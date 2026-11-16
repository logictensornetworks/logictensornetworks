"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const retryRunnerStartup = require("./retry-runner-startup.cjs");

const SHA = "a3261821241f0093346b7f6bf641d9592baa20ac";
const REPOSITORY = "logictensornetworks/logictensornetworks";
const FAILURE = {
  annotation_level: "failure",
  message: "The job was not acquired by Runner of type hosted even after multiple attempts",
};

function fixture() {
  const run = {
    id: 43901735333,
    name: "tests",
    path: ".github/workflows/tests.yml",
    repository: { full_name: REPOSITORY },
    status: "completed",
    conclusion: "failure",
    run_attempt: 1,
    run_number: 58,
    workflow_id: 405416572,
    event: "pull_request",
    head_sha: SHA,
    head_branch: "feat/tptp-parser",
    pull_requests: [{ number: 51 }],
  };
  const job = {
    id: 133419835485,
    status: "completed",
    conclusion: "cancelled",
    labels: ["macos-latest"],
    runner_id: 0,
    runner_name: "",
    steps: [],
    check_run_url: `https://api.github.com/repos/${REPOSITORY}/check-runs/133419835485`,
  };
  const state = {
    run,
    job,
    jobs: [{ status: "completed", conclusion: "success" }, job],
    runs: [run],
    annotations: [FAILURE],
    pr: { state: "open", head: { sha: SHA } },
    branch: { commit: { sha: SHA } },
    onFreshRead: () => {},
    calls: [],
    retries: [],
  };
  let runReads = 0;
  const actions = {
    getWorkflowRun: async (args) => {
      state.calls.push(["getWorkflowRun", args]);
      if (++runReads === 2) state.onFreshRead();
      return { data: structuredClone(state.run) };
    },
    listWorkflowRuns: Symbol("listWorkflowRuns"),
    listJobsForWorkflowRun: Symbol("listJobsForWorkflowRun"),
    reRunWorkflowFailedJobs: async (args) => {
      state.calls.push(["reRunWorkflowFailedJobs", args]);
      state.retries.push(args);
    },
  };
  const checks = { listAnnotations: Symbol("listAnnotations") };
  state.github = {
    rest: {
      actions,
      checks,
      pulls: {
        get: async (args) => {
          state.calls.push(["getPull", args]);
          return { data: structuredClone(state.pr) };
        },
      },
      repos: {
        getBranch: async (args) => {
          state.calls.push(["getBranch", args]);
          return { data: structuredClone(state.branch) };
        },
      },
    },
    paginate: async (endpoint, args) => {
      state.calls.push([endpoint.description, args]);
      if (endpoint === actions.listWorkflowRuns) return structuredClone(state.runs);
      if (endpoint === actions.listJobsForWorkflowRun) return structuredClone(state.jobs);
      if (endpoint === checks.listAnnotations) return structuredClone(state.annotations);
      assert.fail("Unexpected API endpoint");
    },
  };
  state.context = {
    eventName: "workflow_run",
    repo: { owner: "logictensornetworks", repo: "logictensornetworks" },
    payload: {
      repository: { full_name: REPOSITORY, default_branch: "master" },
      workflow_run: structuredClone(run),
    },
  };
  state.core = { info() {}, notice() {} };
  state.recover = () => retryRunnerStartup(state);
  return state;
}

test("recovers the recorded macOS provisioning failure without rerunning successes", async () => {
  const f = fixture();
  const result = await f.recover();
  assert.equal(result.retried, true);
  assert.deepEqual(f.retries, [{
    owner: "logictensornetworks", repo: "logictensornetworks", run_id: f.run.id,
  }]);
  assert.equal(f.calls.filter(([name]) => name === "getWorkflowRun").length, 2);
  assert.ok(f.calls.some(([name, args]) => name === "listJobsForWorkflowRun" &&
    args.filter === "latest" && args.per_page === 100));
  assert.ok(f.calls.some(([name, args]) => name === "listAnnotations" &&
    args.check_run_id === f.job.id && args.per_page === 100));
});

test("handles multiple startup failures across paginated metadata", async () => {
  const f = fixture();
  // paginate returns the combined pages, including jobs after the first 100 successes.
  f.jobs.unshift(...Array.from({ length: 100 }, () => ({
    status: "completed", conclusion: "success",
  })));
  f.jobs.push({ ...f.job, id: 123, check_run_url: "https://api.github.com/x/check-runs/123" });
  assert.equal((await f.recover()).retried, true);
  assert.equal(f.calls.filter(([name]) => name === "listAnnotations").length, 2);
  assert.equal(f.retries.length, 1);
});

test("keeps real failures, manual cancellations, and unknown startup failures visible", async (t) => {
  const cases = {
    "actual test failure": (f) => { f.job.conclusion = "failure"; },
    "install step ran": (f) => { f.job.steps = [{ name: "Install dependencies" }]; },
    "runner was assigned": (f) => { f.job.runner_id = 5; f.job.runner_name = "runner"; },
    "runner name was assigned": (f) => { f.job.runner_name = "runner"; },
    "runner ID missing": (f) => { delete f.job.runner_id; },
    "step metadata missing": (f) => { delete f.job.steps; },
    "step metadata null": (f) => { f.job.steps = null; },
    "Linux job": (f) => { f.job.labels = ["ubuntu-latest"]; },
    "self-hosted macOS": (f) => { f.job.labels = ["self-hosted", "macos-15"]; },
    "no labels": (f) => { f.job.labels = []; },
    "capacity warning alone": (f) => {
      f.annotations = [{ annotation_level: "notice", message: "Due to capacity constraints" }];
    },
    "generic internal server error": (f) => {
      f.annotations = [{ annotation_level: "failure", message: "Internal server error" }];
    },
    "startup notice without error": (f) => {
      f.annotations = [{ ...FAILURE, annotation_level: "notice" }];
    },
    "manual cancellation annotation": (f) => {
      f.annotations = [{ annotation_level: "failure", message: "The operation was canceled." }];
    },
    "no annotation": (f) => { f.annotations = []; },
    "missing check URL": (f) => { delete f.job.check_run_url; },
    "invalid check ID": (f) => { f.job.check_run_url = "https://api.github.com/x/check-runs/0"; },
    "malformed check ID": (f) => { f.job.check_run_url = "https://api.github.com/x/check-runs/text"; },
    "a second job failed tests": (f) => {
      f.jobs.push({ ...f.job, conclusion: "failure", steps: [{ name: "Run tests" }] });
    },
    "another job is active": (f) => {
      f.jobs.push({ status: "in_progress", conclusion: null });
    },
    "no jobs": (f) => { f.jobs = []; },
    "no unsuccessful jobs": (f) => { f.jobs = [{ status: "completed", conclusion: "success" }]; },
  };
  for (const [name, mutate] of Object.entries(cases)) {
    await t.test(name, async () => {
      const f = fixture();
      mutate(f);
      assert.equal((await f.recover()).retried, false);
      assert.deepEqual(f.retries, []);
    });
  }
});

test("rejects unrelated events, workflows, repositories, and stale or repeated runs", async (t) => {
  const cases = {
    "wrong event": (f) => { f.context.eventName = "workflow_dispatch"; },
    "missing event run": (f) => { delete f.context.payload.workflow_run; },
    "wrong event repository": (f) => { f.context.payload.repository.full_name = "other/repo"; },
    "wrong run repository": (f) => { f.run.repository.full_name = "other/repo"; },
    "wrong workflow name": (f) => { f.run.name = "deploy"; },
    "wrong workflow path": (f) => { f.run.path = ".github/workflows/deploy.yml"; },
    "run not completed": (f) => { f.run.status = "in_progress"; },
    "workflow cancelled manually": (f) => { f.run.conclusion = "cancelled"; },
    "workflow succeeded": (f) => { f.run.conclusion = "success"; },
    "retry budget used": (f) => { f.run.run_attempt = 2; },
    "stale event attempt": (f) => { f.context.payload.workflow_run.run_attempt = 2; },
    "event head differs": (f) => { f.context.payload.workflow_run.head_sha = "b".repeat(40); },
    "PR head changed": (f) => { f.pr.head.sha = "b".repeat(40); },
    "closed PR": (f) => { f.pr.state = "closed"; },
    "no associated PR": (f) => { f.run.pull_requests = []; },
    "multiple associated PRs": (f) => { f.run.pull_requests.push({ number: 52 }); },
    "unsupported run event": (f) => { f.run.event = "pull_request_target"; },
    "newer same-SHA PR run exists": (f) => {
      f.runs.push({ ...f.run, id: f.run.id + 1, run_number: f.run.run_number + 1 });
    },
  };
  for (const [name, mutate] of Object.entries(cases)) {
    await t.test(name, async () => {
      const f = fixture();
      mutate(f);
      assert.equal((await f.recover()).retried, false);
      assert.deepEqual(f.retries, []);
    });
  }
});

test("skipped and neutral jobs do not block a genuine startup recovery", async () => {
  const f = fixture();
  f.jobs.push({ status: "completed", conclusion: "skipped" });
  f.jobs.push({ status: "completed", conclusion: "neutral" });
  assert.equal((await f.recover()).retried, true);
});

test("a newer run for a different PR does not supersede this PR", async () => {
  const f = fixture();
  f.runs.push({ ...f.run, run_number: f.run.run_number + 1, pull_requests: [{ number: 52 }] });
  assert.equal((await f.recover()).retried, true);
});

test("recovers only the current default branch for non-PR events", async (t) => {
  for (const event of ["push", "schedule", "workflow_dispatch"]) {
    await t.test(event, async () => {
      const f = fixture();
      f.run.event = event;
      f.run.head_branch = "master";
      assert.equal((await f.recover()).retried, true);
      assert.equal(f.calls.some(([name]) => name === "getPull"), false);
    });
  }
  for (const change of ["branch", "head", "newer run"]) {
    await t.test(`rejects stale ${change}`, async () => {
      const f = fixture();
      f.run.event = "push";
      f.run.head_branch = "master";
      if (change === "branch") f.run.head_branch = "feature";
      if (change === "head") f.branch.commit.sha = "b".repeat(40);
      if (change === "newer run") f.runs.push({ ...f.run, run_number: f.run.run_number + 1 });
      assert.equal((await f.recover()).retried, false);
      assert.deepEqual(f.retries, []);
    });
  }
});

test("rechecks fresh run, head, and same-SHA supersession before requesting recovery", async (t) => {
  const cases = {
    "manual retry started": (f) => { f.run.run_attempt = 2; },
    "run is active": (f) => { f.run.status = "in_progress"; },
    "new PR commit": (f) => { f.pr.head.sha = "b".repeat(40); },
    "closed PR": (f) => { f.pr.state = "closed"; },
    "newer same-SHA run": (f) => {
      f.runs.push({ ...f.run, run_number: f.run.run_number + 1 });
    },
  };
  for (const [name, mutate] of Object.entries(cases)) {
    await t.test(name, async () => {
      const f = fixture();
      f.onFreshRead = () => mutate(f);
      assert.equal((await f.recover()).retried, false);
      assert.deepEqual(f.retries, []);
    });
  }
});

test("API errors remain visible and never cause an uncontrolled retry", async () => {
  const f = fixture();
  f.github.rest.pulls.get = async () => { throw new Error("GitHub API unavailable"); };
  await assert.rejects(f.recover(), /GitHub API unavailable/);
  assert.deepEqual(f.retries, []);
});
