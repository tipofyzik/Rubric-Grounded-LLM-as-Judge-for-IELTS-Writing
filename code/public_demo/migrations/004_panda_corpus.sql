-- Run once in the Supabase SQL Editor. Only PANDA-owned objects are created.
begin;
create schema if not exists extensions;
create extension if not exists vector with schema extensions;
create extension if not exists pgcrypto with schema extensions;
do $$
begin
  if exists (select 1 from pg_catalog.pg_extension e join pg_catalog.pg_namespace n on n.oid=e.extnamespace
             where e.extname in ('vector','pgcrypto') and n.nspname <> 'extensions') then
    raise exception 'vector and pgcrypto must be in extensions; inspect extension schemas before applying this migration';
  end if;
end $$;
create schema if not exists ielts;
grant usage on schema ielts, extensions to anon, authenticated, service_role;

create table if not exists ielts.corpus_releases (
  corpus_id text primary key check (corpus_id ~ '^[0-9a-f]{64}$'),
  manifest jsonb not null,
  state text not null default 'draft' check (state in ('draft','ready')),
  created_at timestamptz not null default now()
);
create table if not exists ielts.reference_essays (
  corpus_id text not null references ielts.corpus_releases(corpus_id),
  essay_id text not null check (length(btrim(essay_id)) > 0),
  essay_hash text not null check (essay_hash ~ '^[0-9a-f]{64}$'),
  task_prompt text not null check (length(btrim(task_prompt)) > 0),
  essay text not null check (length(btrim(essay)) > 0),
  dataset_source text not null check (dataset_source in ('chillies','writing9')),
  split text not null check (split = 'train'),
  source_scores_json text not null,
  embedding extensions.vector(768) not null check (extensions.vector_norm(embedding) between 0.9999 and 1.0001),
  original_tokens integer not null check (original_tokens > 0),
  used_tokens integer not null check (used_tokens > 0 and used_tokens <= original_tokens),
  truncated boolean not null check (truncated = (original_tokens > used_tokens)),
  record_sha256 text not null check (record_sha256 ~ '^[0-9a-f]{64}$'),
  primary key (corpus_id, essay_id), unique (corpus_id, essay_hash)
);
create table if not exists ielts.reference_criteria (
  corpus_id text not null,
  essay_id text not null,
  criterion text not null check (criterion in ('task_response','coherence_and_cohesion','lexical_resource','grammatical_range_and_accuracy')),
  score smallint not null check (score between 0 and 9),
  score_source text not null check (score_source = 'source_dataset_label'),
  insight jsonb not null check (jsonb_typeof(insight) = 'object'),
  primary key (corpus_id, essay_id, criterion),
  foreign key (corpus_id, essay_id) references ielts.reference_essays(corpus_id, essay_id)
);
create table if not exists ielts.rubric_passages (
  rubric_sha256 text not null, criterion text not null, descriptor jsonb not null,
  source text not null, primary key (rubric_sha256, criterion)
);

alter table ielts.corpus_releases enable row level security;
alter table ielts.reference_essays enable row level security;
alter table ielts.reference_criteria enable row level security;
alter table ielts.rubric_passages enable row level security;
drop policy if exists ready_releases on ielts.corpus_releases;
create policy ready_releases on ielts.corpus_releases for select to anon, authenticated using (state='ready');
drop policy if exists ready_essays on ielts.reference_essays;
create policy ready_essays on ielts.reference_essays for select to anon, authenticated using
  (exists(select 1 from ielts.corpus_releases c where c.corpus_id=reference_essays.corpus_id and c.state='ready'));
drop policy if exists ready_criteria on ielts.reference_criteria;
create policy ready_criteria on ielts.reference_criteria for select to anon, authenticated using
  (exists(select 1 from ielts.corpus_releases c where c.corpus_id=reference_criteria.corpus_id and c.state='ready'));
drop policy if exists ready_rubrics on ielts.rubric_passages;
create policy ready_rubrics on ielts.rubric_passages for select to anon, authenticated using
  (exists(select 1 from ielts.corpus_releases c where c.state='ready' and c.manifest->'generator'->>'rubric_sha256'=rubric_passages.rubric_sha256));
revoke all on ielts.corpus_releases, ielts.reference_essays, ielts.reference_criteria, ielts.rubric_passages from public, anon, authenticated;
grant select on ielts.corpus_releases, ielts.reference_essays, ielts.reference_criteria, ielts.rubric_passages to anon, authenticated;
grant all on ielts.corpus_releases, ielts.reference_essays, ielts.reference_criteria, ielts.rubric_passages to service_role;

create or replace function public.panda_begin(p_manifest jsonb) returns jsonb
language plpgsql security invoker set search_path = pg_catalog, extensions as $$
declare c ielts.corpus_releases; m jsonb := p_manifest; k text; item jsonb;
begin
  if m->>'schema_version' is distinct from 'panda-corpus-v1'
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
    insert into ielts.reference_essays values(p_corpus_id,r->>'essay_id',r->>'essay_hash',r->>'task_prompt',r->>'essay',
      r->>'dataset_source',r->>'split',r->>'source_scores_json',((r->'embedding')::text)::extensions.vector,
      (r->>'original_tokens')::integer,(r->>'used_tokens')::integer,(r->>'truncated')::boolean,h);
    for k,item in select key,value from jsonb_each(r->'criteria') loop
      if (item->>'score') is null or (item->>'score') !~ '^[0-9]$' or jsonb_typeof(item->'score') <> 'number'
         or item->>'score_source' is distinct from 'source_dataset_label'
         or jsonb_typeof(item->'evidence') is distinct from 'string' or length(btrim(item->>'evidence'))=0
         or position((item->>'evidence') in (r->>'essay'))=0
         or jsonb_typeof(item->'why_not_higher') is distinct from 'string' or length(btrim(item->>'why_not_higher'))=0
         or jsonb_typeof(item->'why_not_lower') is distinct from 'string' or length(btrim(item->>'why_not_lower'))=0
         or ((item->>'why_not_higher'='not_applicable') is distinct from (item->>'score'='9'))
         or ((item->>'why_not_lower'='not_applicable') is distinct from (item->>'score'='0'))
         or lower(regexp_replace(item->>'evidence','[[:space:]_/-]+','','g')) ~ '^(notapplicable|na|none|null)[.!]?$'
         or (item->>'score'<>'9' and lower(regexp_replace(item->>'why_not_higher','[[:space:]_/-]+','','g')) ~ '^(notapplicable|na|none|null)[.!]?$')
         or (item->>'score'<>'0' and lower(regexp_replace(item->>'why_not_lower','[[:space:]_/-]+','','g')) ~ '^(notapplicable|na|none|null)[.!]?$') then
        raise exception 'Invalid criterion insight/score: %',k;
      end if;
      insert into ielts.reference_criteria values(p_corpus_id,r->>'essay_id',k,(item->>'score')::smallint,
        item->>'score_source',item - 'score' - 'score_source');
    end loop;
  end loop;
  return jsonb_build_object('count',jsonb_array_length(p_records));
end $$;

create or replace function public.panda_finalize(p_corpus_id text) returns jsonb
language plpgsql security invoker set search_path = pg_catalog, extensions as $$
declare c ielts.corpus_releases; n bigint; criteria_count bigint; actual_digest text;
begin
  select * into strict c from ielts.corpus_releases where corpus_id=p_corpus_id for update;
  select count(*),encode(extensions.digest(convert_to(string_agg(essay_id||':'||record_sha256||E'\n',''
      order by essay_id collate "C"),'UTF8'),'sha256'),'hex') into n,actual_digest
    from ielts.reference_essays where corpus_id=p_corpus_id;
  select count(*) into criteria_count from ielts.reference_criteria where corpus_id=p_corpus_id;
  if n <> (c.manifest->>'count')::bigint or criteria_count <> n*4 or actual_digest is distinct from c.manifest->>'records_digest' then
    raise exception 'Corpus incomplete or membership/content digest differs';
  end if;
  update ielts.corpus_releases set state='ready' where corpus_id=p_corpus_id;
  return jsonb_build_object('corpus_id',p_corpus_id,'state','ready','count',n,'records_digest',actual_digest);
end $$;

create or replace function public.panda_match(
  p_corpus_id text, p_embedding_profile_sha256 text, p_query_embedding jsonb,
  p_query_hash text, p_query_id text default null, p_match_count integer default 3, p_threshold real default null
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
      'scores',a.scores,'insights',a.insights) order by n.similarity desc,n.essay_id collate "C"),'[]'::jsonb)
    into matches from nearest n cross join lateral (
      select jsonb_object_agg(criterion,score) as scores,jsonb_object_agg(criterion,insight) as insights
      from ielts.reference_criteria rc where rc.corpus_id=n.corpus_id and rc.essay_id=n.essay_id
    ) a;
  return jsonb_build_object('corpus_id',p_corpus_id,'embedding_profile_sha256',p_embedding_profile_sha256,
    'count',(c.manifest->>'count')::integer,'references',matches);
end $$;

revoke all on function public.panda_begin(jsonb) from public,anon,authenticated;
revoke all on function public.panda_import(text,jsonb) from public,anon,authenticated;
revoke all on function public.panda_finalize(text) from public,anon,authenticated;
revoke all on function public.panda_match(text,text,jsonb,text,text,integer,real) from public,anon,authenticated;
grant execute on function public.panda_begin(jsonb),public.panda_import(text,jsonb),public.panda_finalize(text) to service_role;
grant execute on function public.panda_match(text,text,jsonb,text,text,integer,real) to anon,authenticated,service_role;
notify pgrst, 'reload schema';
commit;
