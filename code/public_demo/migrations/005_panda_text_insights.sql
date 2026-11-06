-- Apply after 004. Preserve complete source responses, including failed/partial text.
-- v1 releases retain their exact-evidence contract; all new builds write v2.
begin;
alter table ielts.reference_essays add column if not exists insights_text text;
alter table ielts.reference_essays add column if not exists generation_status text;
alter table ielts.reference_essays add column if not exists generation_error text;

create or replace function public.panda_begin(p_manifest jsonb) returns jsonb
language plpgsql security invoker set search_path = pg_catalog, extensions as $$
declare c ielts.corpus_releases; m jsonb := p_manifest; k text; item jsonb;
begin
  if (m->>'schema_version' is null or m->>'schema_version' not in ('panda-corpus-v1','panda-corpus-v2'))
     or (m->>'corpus_id') !~ '^[0-9a-f]{64}$' or m->>'corpus_id' is null
     or (m->>'records_digest') !~ '^[0-9a-f]{64}$' or m->>'records_digest' is null
     or (m->>'embedding_profile_sha256') !~ '^[0-9a-f]{64}$' or m->>'embedding_profile_sha256' is null
     or m->'embedding'->>'dimensions' is distinct from '768'
     or m->'embedding'->>'input' is distinct from 'essay'
     or m->'embedding'->>'normalize' is distinct from 'true'
     or m->'train'->>'policy' is distinct from 'all_train'
     or m->>'count' is null or jsonb_typeof(m->'count') is distinct from 'number'
     or (m->>'count')::integer < 1
     or (m->>'count')::integer is distinct from (m->'train'->>'count')::integer then
    raise exception 'Invalid PANDA corpus manifest';
  end if;
  insert into ielts.corpus_releases(corpus_id,manifest) values(m->>'corpus_id',m) on conflict do nothing;
  select * into strict c from ielts.corpus_releases where corpus_id=m->>'corpus_id' for update;
  if c.manifest <> m then raise exception 'Corpus ID already registered with different metadata'; end if;
  for k,item in select key,value from jsonb_each(m->'generator'->'rubric') loop
    insert into ielts.rubric_passages values(m->'generator'->>'rubric_sha256',
      case when k='task_achievement' then 'task_response' else k end,item,m->'generator'->>'rubric_source') on conflict do nothing;
    if not exists(select 1 from ielts.rubric_passages where rubric_sha256=m->'generator'->>'rubric_sha256'
      and criterion=case when k='task_achievement' then 'task_response' else k end and descriptor=item) then
      raise exception 'Rubric revision conflict';
    end if;
  end loop;
  return jsonb_build_object('corpus_id',c.corpus_id,'state',c.state);
end $$;

create or replace function public.panda_import(p_corpus_id text, p_records jsonb) returns jsonb
language plpgsql security invoker set search_path = pg_catalog, extensions as $$
declare c ielts.corpus_releases; raw text; r jsonb; h text; old_hash text; k text; item jsonb;
begin
  select * into strict c from ielts.corpus_releases where corpus_id=p_corpus_id for update;
  if jsonb_typeof(p_records) is distinct from 'array' or jsonb_array_length(p_records) not between 1 and 100 then
    raise exception 'Import batch must contain 1-100 canonical JSON strings';
  end if;
  for raw in select jsonb_array_elements_text(p_records) loop
    r := raw::jsonb;
    h := encode(extensions.digest(convert_to(raw,'UTF8'),'sha256'),'hex');
    select record_sha256 into old_hash from ielts.reference_essays where corpus_id=p_corpus_id and essay_id=r->>'essay_id';
    if found then
      if old_hash <> h then raise exception 'Immutable reference differs: %',r->>'essay_id'; end if;
      continue;
    end if;
    if c.state <> 'draft' then raise exception 'A ready corpus cannot be extended'; end if;
    if jsonb_typeof(r->'criteria') <> 'object' or (select count(*) from jsonb_object_keys(r->'criteria')) <> 4
       or (r->>'used_tokens')::integer <> least((r->>'original_tokens')::integer,(c.manifest->'embedding'->>'max_length')::integer) then
      raise exception 'Invalid reference criteria/lengths';
    end if;
    if c.manifest->>'schema_version' = 'panda-corpus-v2' and (
       jsonb_typeof(r->'insights') is distinct from 'string'
       or r->>'generation_status' is null or r->>'generation_status' not in ('accepted','failed')
       or (r->>'generation_status'='accepted' and r->'generation_error' is distinct from 'null'::jsonb)
       or (r->>'generation_status'='failed' and (jsonb_typeof(r->'generation_error') is distinct from 'string'
           or length(btrim(r->>'generation_error'))=0))) then
      raise exception 'Invalid raw insights/provenance';
    end if;
    insert into ielts.reference_essays(
      corpus_id,essay_id,essay_hash,task_prompt,essay,dataset_source,split,source_scores_json,
      embedding,original_tokens,used_tokens,truncated,record_sha256,insights_text,generation_status,generation_error)
    values(p_corpus_id,r->>'essay_id',r->>'essay_hash',r->>'task_prompt',r->>'essay',
      r->>'dataset_source',r->>'split',r->>'source_scores_json',((r->'embedding')::text)::extensions.vector,
      (r->>'original_tokens')::integer,(r->>'used_tokens')::integer,(r->>'truncated')::boolean,h,
      case when c.manifest->>'schema_version'='panda-corpus-v2' then r->>'insights' end,
      case when c.manifest->>'schema_version'='panda-corpus-v2' then r->>'generation_status' end,
      case when c.manifest->>'schema_version'='panda-corpus-v2' then r->>'generation_error' end);
    for k,item in select key,value from jsonb_each(r->'criteria') loop
      if (item->>'score') is null or (item->>'score') !~ '^[0-9]$' or jsonb_typeof(item->'score') <> 'number'
         or item->>'score_source' is distinct from 'source_dataset_label'
         or (c.manifest->>'schema_version'='panda-corpus-v1' and (
            jsonb_typeof(item->'evidence') is distinct from 'string' or length(btrim(item->>'evidence'))=0
         or position((item->>'evidence') in (r->>'essay'))=0
         or jsonb_typeof(item->'why_not_higher') is distinct from 'string' or length(btrim(item->>'why_not_higher'))=0
         or jsonb_typeof(item->'why_not_lower') is distinct from 'string' or length(btrim(item->>'why_not_lower'))=0
         or ((item->>'why_not_higher'='not_applicable') is distinct from (item->>'score'='9'))
         or ((item->>'why_not_lower'='not_applicable') is distinct from (item->>'score'='0'))
         or lower(regexp_replace(item->>'evidence','[[:space:]_/-]+','','g')) ~ '^(notapplicable|na|none|null)[.!]?$'
         or (item->>'score'<>'9' and lower(regexp_replace(item->>'why_not_higher','[[:space:]_/-]+','','g')) ~ '^(notapplicable|na|none|null)[.!]?$')
         or (item->>'score'<>'0' and lower(regexp_replace(item->>'why_not_lower','[[:space:]_/-]+','','g')) ~ '^(notapplicable|na|none|null)[.!]?$')))
         or (c.manifest->>'schema_version'='panda-corpus-v2' and item - 'score' - 'score_source' <> '{}'::jsonb) then
        raise exception 'Invalid criterion insight/score: %',k;
      end if;
      insert into ielts.reference_criteria values(p_corpus_id,r->>'essay_id',k,(item->>'score')::smallint,
        item->>'score_source',item - 'score' - 'score_source');
    end loop;
  end loop;
  return jsonb_build_object('count',jsonb_array_length(p_records));
end $$;

create or replace function public.panda_match(
  p_corpus_id text, p_embedding_profile_sha256 text, p_query_embedding jsonb,
  p_query_hash text, p_query_id text default null, p_match_count integer default 1, p_threshold real default null
) returns jsonb language plpgsql stable security invoker set search_path = pg_catalog, extensions as $$
declare c ielts.corpus_releases; q extensions.vector; matches jsonb;
begin
  select * into c from ielts.corpus_releases where corpus_id=p_corpus_id and state='ready';
  if not found then raise exception 'Ready corpus unavailable'; end if;
  if c.manifest->>'embedding_profile_sha256' is distinct from p_embedding_profile_sha256 then raise exception 'Embedding profile mismatch'; end if;
  if p_match_count is null or p_match_count not between 1 and 20 or p_query_hash is null or p_query_hash !~ '^[0-9a-f]{64}$'
     or p_threshold < -1 or p_threshold > 1 or p_threshold='NaN'::real
     or jsonb_typeof(p_query_embedding) is distinct from 'array' then raise exception 'Invalid search parameters'; end if;
  q := (p_query_embedding::text)::extensions.vector;
  if extensions.vector_dims(q) <> 768 or extensions.vector_norm(q) not between 0.9999 and 1.0001 then raise exception 'Invalid query vector'; end if;
  with nearest as (
    select e.*, 1-(e.embedding operator(extensions.<=>) q) as similarity
    from ielts.reference_essays e
    where e.corpus_id=p_corpus_id and e.essay_hash<>p_query_hash
      and (p_query_id is null or e.essay_id<>p_query_id)
      and (p_threshold is null or 1-(e.embedding operator(extensions.<=>) q)>=p_threshold)
    order by e.embedding operator(extensions.<=>) q, e.essay_id collate "C" limit p_match_count
  ) select coalesce(jsonb_agg(jsonb_build_object('essay_id',n.essay_id,'essay_hash',n.essay_hash,
      'task_prompt',n.task_prompt,'essay',n.essay,'similarity',n.similarity,
      'scores',a.scores,'insights',case when c.manifest->>'schema_version'='panda-corpus-v2'
        then to_jsonb(n.insights_text) else a.insights end,
      'generation_status',n.generation_status,'generation_error',n.generation_error) order by n.similarity desc,n.essay_id collate "C"),'[]'::jsonb)
    into matches from nearest n cross join lateral (
      select jsonb_object_agg(criterion,score) as scores,jsonb_object_agg(criterion,insight) as insights
      from ielts.reference_criteria rc where rc.corpus_id=n.corpus_id and rc.essay_id=n.essay_id
    ) a;
  return jsonb_build_object('corpus_id',p_corpus_id,'embedding_profile_sha256',p_embedding_profile_sha256,
    'count',(c.manifest->>'count')::integer,'references',matches);
end $$;

notify pgrst, 'reload schema';
commit;
