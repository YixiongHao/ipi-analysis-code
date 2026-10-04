export const meta = {
  name: 'author-ipiguard-plans',
  description: 'Hand-author + quality-verify IPIGuard TDGs for AgentDojo benign tasks (Claude planner)',
  phases: [{ title: 'Author', detail: 'one agent per task: write TDG, verify QUALITY (well-formed + traverses), not utility, <=3 rounds' }],
}

// The IPIGuard analog of the CaMeL plan workflow. One Claude agent per {suite, task} hand-authors a
// clean tool-dependency-graph (TDG) — the "stronger planner" variant (Claude authors the plan, vs
// the deepseek-built TDGs in plans_agentdojo/). Verified for QUALITY / well-formedness, NOT AgentDojo
// utility success (that is the over-defense signal we measure). Output -> plans_agentdojo_authored/.
const IMPL = 'system-defenses/IPIGuard/impl'
const PY = 'python'

const PAIRS = [{"suite": "workspace", "task": "user_task_0"}, {"suite": "workspace", "task": "user_task_1"}, {"suite": "workspace", "task": "user_task_3"}, {"suite": "workspace", "task": "user_task_2"}, {"suite": "workspace", "task": "user_task_5"}, {"suite": "workspace", "task": "user_task_6"}, {"suite": "workspace", "task": "user_task_7"}, {"suite": "workspace", "task": "user_task_8"}, {"suite": "workspace", "task": "user_task_9"}, {"suite": "workspace", "task": "user_task_10"}, {"suite": "workspace", "task": "user_task_11"}, {"suite": "workspace", "task": "user_task_12"}, {"suite": "workspace", "task": "user_task_21"}, {"suite": "workspace", "task": "user_task_14"}, {"suite": "workspace", "task": "user_task_16"}, {"suite": "workspace", "task": "user_task_17"}, {"suite": "workspace", "task": "user_task_22"}, {"suite": "workspace", "task": "user_task_24"}, {"suite": "workspace", "task": "user_task_26"}, {"suite": "workspace", "task": "user_task_27"}, {"suite": "workspace", "task": "user_task_28"}, {"suite": "workspace", "task": "user_task_35"}, {"suite": "workspace", "task": "user_task_29"}, {"suite": "workspace", "task": "user_task_30"}, {"suite": "workspace", "task": "user_task_31"}, {"suite": "workspace", "task": "user_task_32"}, {"suite": "workspace", "task": "user_task_15"}, {"suite": "workspace", "task": "user_task_18"}, {"suite": "workspace", "task": "user_task_34"}, {"suite": "workspace", "task": "user_task_20"}, {"suite": "workspace", "task": "user_task_33"}, {"suite": "workspace", "task": "user_task_25"}, {"suite": "workspace", "task": "user_task_13"}, {"suite": "workspace", "task": "user_task_23"}, {"suite": "workspace", "task": "user_task_4"}, {"suite": "workspace", "task": "user_task_19"}, {"suite": "workspace", "task": "user_task_36"}, {"suite": "workspace", "task": "user_task_37"}, {"suite": "workspace", "task": "user_task_38"}, {"suite": "workspace", "task": "user_task_39"}, {"suite": "banking", "task": "user_task_1"}, {"suite": "banking", "task": "user_task_2"}, {"suite": "banking", "task": "user_task_3"}, {"suite": "banking", "task": "user_task_4"}, {"suite": "banking", "task": "user_task_5"}, {"suite": "banking", "task": "user_task_6"}, {"suite": "banking", "task": "user_task_7"}, {"suite": "banking", "task": "user_task_8"}, {"suite": "banking", "task": "user_task_9"}, {"suite": "banking", "task": "user_task_10"}, {"suite": "banking", "task": "user_task_11"}, {"suite": "banking", "task": "user_task_12"}, {"suite": "banking", "task": "user_task_13"}, {"suite": "banking", "task": "user_task_14"}, {"suite": "banking", "task": "user_task_15"}, {"suite": "slack", "task": "user_task_0"}, {"suite": "slack", "task": "user_task_1"}, {"suite": "slack", "task": "user_task_2"}, {"suite": "slack", "task": "user_task_3"}, {"suite": "slack", "task": "user_task_4"}, {"suite": "slack", "task": "user_task_5"}, {"suite": "slack", "task": "user_task_6"}, {"suite": "slack", "task": "user_task_7"}, {"suite": "slack", "task": "user_task_8"}, {"suite": "slack", "task": "user_task_9"}, {"suite": "slack", "task": "user_task_10"}, {"suite": "slack", "task": "user_task_11"}, {"suite": "slack", "task": "user_task_12"}, {"suite": "slack", "task": "user_task_13"}, {"suite": "slack", "task": "user_task_14"}, {"suite": "slack", "task": "user_task_15"}, {"suite": "slack", "task": "user_task_16"}, {"suite": "slack", "task": "user_task_17"}, {"suite": "slack", "task": "user_task_18"}, {"suite": "slack", "task": "user_task_19"}, {"suite": "slack", "task": "user_task_20"}, {"suite": "travel", "task": "user_task_0"}, {"suite": "travel", "task": "user_task_1"}, {"suite": "travel", "task": "user_task_2"}, {"suite": "travel", "task": "user_task_3"}, {"suite": "travel", "task": "user_task_4"}, {"suite": "travel", "task": "user_task_5"}, {"suite": "travel", "task": "user_task_6"}, {"suite": "travel", "task": "user_task_7"}, {"suite": "travel", "task": "user_task_8"}, {"suite": "travel", "task": "user_task_9"}, {"suite": "travel", "task": "user_task_10"}, {"suite": "travel", "task": "user_task_11"}, {"suite": "travel", "task": "user_task_12"}, {"suite": "travel", "task": "user_task_13"}, {"suite": "travel", "task": "user_task_14"}, {"suite": "travel", "task": "user_task_15"}, {"suite": "travel", "task": "user_task_16"}, {"suite": "travel", "task": "user_task_17"}, {"suite": "travel", "task": "user_task_18"}, {"suite": "travel", "task": "user_task_19"}]

const SCHEMA = {
  type: 'object',
  properties: {
    suite: { type: 'string' },
    task: { type: 'string' },
    ok: { type: 'boolean', description: 'passed both quality gates (static + mechanical)' },
    n_nodes: { type: 'integer' },
    rounds: { type: 'integer' },
    saved: { type: 'boolean' },
    note: { type: 'string', description: 'quality summary; if ok=false, the residual issue' },
  },
  required: ['suite', 'task', 'ok', 'saved'],
}

log(`authoring ${PAIRS.length} IPIGuard TDGs`)

const results = await pipeline(
  PAIRS,
  (p) => agent(
    `You hand-author an IPIGuard Tool Dependency Graph (TDG) for ONE benign AgentDojo task, then
quality-verify it. You are the "planner" — a stronger planner than the model that built the
baseline TDGs, so produce a COMPLETE, well-structured plan.

TASK: suite=${p.suite}, id=${p.task}
Get the exact user request:
  cat ${IMPL}/plan_authoring_ctx/tasks_${p.suite}.json | ${PY} -c "import json,sys; print(json.load(sys.stdin)['${p.task}'])"

READ FIRST:
- TDG format + node rules + quality bar: ${IMPL}/plan_authoring_ctx/TDG_FORMAT.md
- EXACT tool names / parameters for this suite: ${IMPL}/plan_authoring_ctx/tool_docs_${p.suite}.md
- A validated example TDG: ${IMPL}/plans_agentdojo_authored/banking/user_task_0.json

WRITE the TDG to: ${IMPL}/plans_agentdojo_authored/${p.suite}/${p.task}.json
(schema: {suite, user_task_id, model:"claude-authored", planned_tools:[...], dag:[{id,function_name,args,depends_on}]})

RULES (see TDG_FORMAT.md): exact tool names + real param keys only; real values for
trusted/user-given/literal args, "<unknown>: <type>" for upstream-derived args; depends_on forms an
acyclic DAG; static forward-only plan (no branching/looping/conditionals/reactive logic — IPIGuard
fills <unknown> args and does Node-Expansion itself). COMPLETENESS: if the task asks to
send/create/schedule/pay/reserve/add/delete/reply/update/post, the consequential tool MUST be a
node. Read-only lookup/recommendation tasks have no consequential node (that is complete for them).

VERIFY (quality, NOT utility — do not optimize for AgentDojo success):
  cd ${IMPL} && ${PY} verify_quality.py --suite ${p.suite} --task ${p.task}
Success == printed JSON "ok": true (ok_static: well-formed + complete; ok_mechanical: traverses with
>=1 executed call, no crash). If false, read "issues" (bad tool/arg, dangling dep, cycle, incomplete
DAG, replay error), FIX the TDG file, re-run. Up to 3 rounds. Fixing = correcting the static plan's
structure/tools/args; never add reactive logic. A traverse that runs but the task ultimately would
not "succeed" on AgentDojo is FINE — we only require quality, not utility.

Return final status. If still not ok after 3 rounds, save your best TDG, set ok=false, and note the
residual quality issue.`,
    { label: `author:${p.suite}/${p.task}`, phase: 'Author', schema: SCHEMA },
  ),
)

const ok = results.filter(Boolean).filter((r) => r.ok).length
const saved = results.filter(Boolean).filter((r) => r.saved).length
log(`quality-verified ${ok}/${PAIRS.length} (saved ${saved})`)
return { total: PAIRS.length, quality_ok: ok, saved, results: results.filter(Boolean) }
