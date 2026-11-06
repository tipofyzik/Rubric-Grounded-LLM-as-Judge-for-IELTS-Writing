// Real PostgreSQL/pgvector integration without a running Docker daemon.
// Install pinned packages in .cache/sql, then pass a fixture Corpus directory.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { createHash } from 'node:crypto';

const require = createRequire(path.resolve('.cache/sql/node_modules/probe.cjs'));
const { PGlite } = require('@electric-sql/pglite');
const { vector } = require('@electric-sql/pglite-pgvector');
const { pgcrypto } = require('@electric-sql/pglite/contrib/pgcrypto');
const root = process.argv[2];
if (!root) throw new Error('Pass a verified fixture Corpus directory');
const manifest = JSON.parse(await readFile(path.join(root,'manifest.json'),'utf8'));
const records = (await readFile(path.join(root,'records.jsonl'),'utf8')).trimEnd().split('\n');
const rows = records.map(JSON.parse);
const rawText = manifest.schema_version === 'panda-corpus-v2';
const pg = new PGlite({ extensions: { vector, pgcrypto } });
const checks = [];
async function check(name, fn) { await fn(); checks.push(name); }
async function rpc(name, parameters) {
  const names = Object.keys(parameters);
  const query = `select public.${name}(${names.map((key,i)=>`${key} => $${i+1}${['p_manifest','p_records','p_query_embedding'].includes(key)?'::jsonb':''}`).join(',')}) as result`;
  const result = await pg.query(query,Object.values(parameters).map(v=>typeof v==='object'&&v!==null?JSON.stringify(v):v));
  return result.rows[0].result;
}
const query = {
  p_corpus_id:manifest.corpus_id, p_embedding_profile_sha256:manifest.embedding_profile_sha256,
  p_query_embedding:rows[0].embedding, p_query_hash:rows[0].essay_hash, p_query_id:null, p_match_count:3, p_threshold:null
};
try {
  await pg.exec('create role anon; create role authenticated; create role service_role bypassrls;');
  const baseSql = await readFile('code/public_demo/migrations/004_panda_corpus.sql','utf8');
  const textSql = await readFile('code/public_demo/migrations/005_panda_text_insights.sql','utf8');
  const sql = baseSql + '\n' + textSql;
  await check('migration and idempotent migration',async()=>{await pg.exec(sql);await pg.exec(sql);});
  await pg.exec('set role anon');
  await check('anon cannot register or write',async()=>{
    await assert.rejects(()=>rpc('panda_begin',{p_manifest:manifest}),/permission denied/);
    await assert.rejects(()=>pg.exec("insert into ielts.corpus_releases values ('x','{}','draft',now())"),/permission denied/);
  });
  await pg.exec('set role service_role');
  await check('missing count and null batch are rejected',async()=>{
    const bad={...manifest}; delete bad.count;
    await assert.rejects(()=>rpc('panda_begin',{p_manifest:bad}),/Invalid/);
  });
  await check('registration and repeated registration',async()=>{
    assert.equal((await rpc('panda_begin',{p_manifest:manifest})).state,'draft');
    assert.equal((await rpc('panda_begin',{p_manifest:manifest})).state,'draft');
  });
  await check('metadata conflict',async()=>{
    await assert.rejects(()=>rpc('panda_begin',{p_manifest:{...manifest,count:manifest.count+1}}));
  });
  await check('incomplete corpus cannot finalize',async()=>{
    await assert.rejects(()=>rpc('panda_finalize',{p_corpus_id:manifest.corpus_id}),/incomplete/);
    await assert.rejects(()=>rpc('panda_import',{p_corpus_id:manifest.corpus_id,p_records:null}),/Import batch/);
  });
  await check('draft hidden from reader',async()=>{
    await pg.exec('set role anon');
    assert.equal((await pg.query('select count(*)::int n from ielts.corpus_releases')).rows[0].n,0);
    await assert.rejects(()=>rpc('panda_match',query),/unavailable/);
    await pg.exec('set role service_role');
  });
  await check('failed batch rolls back prior row in same batch',async()=>{
    const bad=JSON.parse(records[1]); bad.criteria.task_response.score=6.5;
    await assert.rejects(()=>rpc('panda_import',{p_corpus_id:manifest.corpus_id,p_records:[records[0],JSON.stringify(bad)]}),/Invalid criterion/);
    assert.equal((await pg.query('select count(*)::int n from ielts.reference_essays')).rows[0].n,0);
  });
  for (const [name,mutate] of [
    ['test split exclusion',r=>{r.split='test';}],
    ['fraction/bool score rejection',r=>{r.criteria.task_response.score=true;}],
    [rawText?'nontext raw insight rejection':'invented evidence rejection',r=>{
      if(rawText)r.insights={};else r.criteria.task_response.evidence='invented quote';
    }],
    ['zero vector rejection',r=>{r.embedding=r.embedding.map(()=>0);}],
    ['wrong vector dimension rejection',r=>{r.embedding=[1,0];}]
  ]) await check(name,async()=>{
    const bad=JSON.parse(records[0]); mutate(bad);
    await assert.rejects(()=>rpc('panda_import',{p_corpus_id:manifest.corpus_id,p_records:[JSON.stringify(bad)]}));
  });
  if(!rawText)await check('placeholder spelling variants rejected in required explanations',async()=>{
    for(const text of ['not_applicable','Not applicable','not applicable','NOT-APPLICABLE.','N/A','n_a','None','null!']) {
      for(const field of ['why_not_higher','why_not_lower']) {
        const bad=JSON.parse(records[0]); bad.criteria.task_response.score=4;
        bad.criteria.task_response[field]=text;
        await assert.rejects(()=>rpc('panda_import',{p_corpus_id:manifest.corpus_id,p_records:[JSON.stringify(bad)]}),/Invalid criterion/);
      }
    }
  });
  if(!rawText)await check('actual boundary still requires the canonical placeholder',async()=>{
    const bad=JSON.parse(records[0]); bad.criteria.task_response.score=0;
    bad.criteria.task_response.why_not_lower='Not applicable';
    await assert.rejects(()=>rpc('panda_import',{p_corpus_id:manifest.corpus_id,p_records:[JSON.stringify(bad)]}),/Invalid criterion/);
  });
  if(!rawText)await check('canonical placeholders accepted at real zero and nine boundaries',async()=>{
    for(const [score,field] of [[0,'why_not_lower'],[9,'why_not_higher']]) {
      await pg.exec('begin');
      try {
        const record=JSON.parse(records[0]); record.criteria.task_response.score=score;
        record.criteria.task_response[field]='not_applicable';
        assert.equal((await rpc('panda_import',{p_corpus_id:manifest.corpus_id,p_records:[JSON.stringify(record)]})).count,1);
      } finally { await pg.exec('rollback'); }
    }
  });
  if(rawText)await check('failed partial text roundtrips verbatim without a parser',async()=>{
    await pg.exec('begin');
    try {
      const record=JSON.parse(records[0]);
      record.insights='## Task Response\n- Evidence: Descriptive text, with "quotes".\n\nUnfinished —';
      record.generation_status='failed';record.generation_error='Generation did not finish with EOS';
      await rpc('panda_import',{p_corpus_id:manifest.corpus_id,p_records:[JSON.stringify(record)]});
      const stored=(await pg.query('select insights_text,generation_status,generation_error from ielts.reference_essays')).rows[0];
      assert.equal(stored.insights_text,record.insights);
      assert.equal(stored.generation_status,'failed');
      assert.equal(stored.generation_error,record.generation_error);
    } finally {await pg.exec('rollback');}
  });
  await check('shuffled import, repeat, content conflict',async()=>{
    const reversed=[...records].reverse();
    assert.equal((await rpc('panda_import',{p_corpus_id:manifest.corpus_id,p_records:reversed})).count,records.length);
    assert.equal((await rpc('panda_import',{p_corpus_id:manifest.corpus_id,p_records:records})).count,records.length);
    const bad=JSON.parse(records[0]); bad.task_prompt='Changed task';
    await assert.rejects(()=>rpc('panda_import',{p_corpus_id:manifest.corpus_id,p_records:[JSON.stringify(bad)]}),/Immutable/);
  });
  await check('finalization checks Python/SQL digest',async()=>{
    const result=await rpc('panda_finalize',{p_corpus_id:manifest.corpus_id});
    assert.equal(result.state,'ready'); assert.equal(result.records_digest,manifest.records_digest);
    assert.equal((await rpc('panda_finalize',{p_corpus_id:manifest.corpus_id})).state,'ready');
  });
  await pg.exec('set role anon');
  await check('self hash excluded under new ID and criteria remain aligned',async()=>{
    const result=await rpc('panda_match',{...query,p_query_id:'new-source-id'});
    assert.equal(result.references.length,Math.min(3,rows.length-1));
    assert.ok(!result.references.some(r=>r.essay_hash===rows[0].essay_hash));
    for(const found of result.references) {
      const original=rows.find(r=>r.essay_id===found.essay_id);
      assert.equal(found.essay,original.essay); assert.equal(found.task_prompt,original.task_prompt);
      for(const [key,value] of Object.entries(original.criteria)) {
        assert.equal(found.scores[key],value.score);
        if(!rawText)assert.equal(found.insights[key].evidence,value.evidence);
      }
      if(rawText) {
        assert.equal(found.insights,original.insights);
        assert.equal(found.generation_status,original.generation_status);
        assert.equal(found.generation_error,original.generation_error);
      }
    }
  });
  await check('default one reference and explicit configurable count',async()=>{
    const defaults={...query};delete defaults.p_match_count;
    assert.equal((await rpc('panda_match',defaults)).references.length,1);
    assert.equal((await rpc('panda_match',{...query,p_match_count:2})).references.length,2);
  });
  await check('exact cosine order vs independent JavaScript calculation',async()=>{
    const input=rows[0].embedding.map((v,i)=>v+0.5*rows[1].embedding[i]);
    const norm=Math.hypot(...input); const q=input.map(v=>v/norm);
    const result=await rpc('panda_match',{...query,p_query_embedding:q,p_query_hash:'f'.repeat(64)});
    const expected=rows.map(r=>({id:r.essay_id,s:r.embedding.reduce((n,v,i)=>n+v*q[i],0)/Math.hypot(...r.embedding)}))
      .sort((a,b)=>b.s-a.s||(a.id<b.id?-1:1)).slice(0,3);
    assert.deepEqual(result.references.map(r=>r.essay_id),expected.map(r=>r.id));
    result.references.forEach((r,i)=>assert.ok(Math.abs(r.similarity-expected[i].s)<1e-5));
  });
  await check('empty differs from release/profile errors',async()=>{
    const result=await rpc('panda_match',{...query,p_threshold:1}); assert.deepEqual(result.references,[]);
    await assert.rejects(()=>rpc('panda_match',{...query,p_embedding_profile_sha256:'b'.repeat(64)}),/profile mismatch/);
    await assert.rejects(()=>rpc('panda_match',{...query,p_corpus_id:'c'.repeat(64)}),/unavailable/);
  });
  await check('search parameter guards',async()=>{
    for(const values of [{p_match_count:0},{p_match_count:21},{p_threshold:2},{p_query_embedding:[0,1]},{p_query_embedding:Array(768).fill(0)}])
      await assert.rejects(()=>rpc('panda_match',{...query,...values}));
  });
  await check('ready corpus remains readable after failures and writes forbidden',async()=>{
    assert.equal((await rpc('panda_match',query)).references.length,3);
    await assert.rejects(()=>rpc('panda_import',{p_corpus_id:manifest.corpus_id,p_records:records}),/permission denied/);
  });
  const versions=(await pg.query("select version(), (select extversion from pg_extension where extname='vector') as pgvector")).rows[0];
  console.log(JSON.stringify({status:'passed',cases:checks.length,checks,versions,migration_sha256:createHash('sha256').update(sql).digest('hex')},null,2));
} catch(error) {
  console.error(JSON.stringify({status:'failed',after:checks.at(-1),message:error.message},null,2));
  process.exitCode=1;
} finally { await pg.close(); }
