"use strict";

const STARTUP_FAILURE =
  "the job was not acquired by runner of type hosted even after multiple attempts";

/** Retry one failed tests run only when GitHub never started its macOS jobs. */
module.exports = async function retryRunnerStartup({ github, context, core }) {
  const eventRun = context.payload.workflow_run;
  const repository = context.payload.repository;
  const { owner, repo } = context.repo;
  const repositoryName = `${owner}/${repo}`;

  function skip(reason) {
    core.info(reason);
    return { retried: false, reason };
  }

  if (
    context.eventName !== "workflow_run" ||
    !Number.isSafeInteger(eventRun?.id) ||
    eventRun.id <= 0 ||
    repository?.full_name !== repositoryName
  ) {
    return skip("Ignoring an event without a trusted workflow run and repository.");
  }

  function eligibleRun(run) {
    return (
      run.id === eventRun.id &&
      run.repository?.full_name === repositoryName &&
      run.name === "tests" &&
      run.path === ".github/workflows/tests.yml" &&
      Number.isSafeInteger(run.workflow_id) &&
      run.workflow_id > 0 &&
      Number.isSafeInteger(run.run_number) &&
      run.run_number > 0 &&
      /^[a-f0-9]{40}$/.test(run.head_sha) &&
      eventRun.head_sha === run.head_sha &&
      run.status === "completed" &&
      run.conclusion === "failure" &&
      run.run_attempt === 1 &&
      run.run_attempt === eventRun.run_attempt
    );
  }

  async function currentHead(run) {
    let prNumber;
    if (run.event === "pull_request") {
      if (run.pull_requests?.length !== 1) return false;
      prNumber = run.pull_requests[0].number;
      if (!Number.isSafeInteger(prNumber) || prNumber <= 0) return false;
      const { data: pr } = await github.rest.pulls.get({ owner, repo, pull_number: prNumber });
      if (pr.state !== "open" || pr.head.sha !== run.head_sha) return false;
    } else {
      if (
        !["push", "schedule", "workflow_dispatch"].includes(run.event) ||
        !repository.default_branch ||
        run.head_branch !== repository.default_branch
      ) {
        return false;
      }
      const { data: branch } = await github.rest.repos.getBranch({
        owner,
        repo,
        branch: repository.default_branch,
      });
      if (branch.commit.sha !== run.head_sha) return false;
    }

    const runs = await github.paginate(github.rest.actions.listWorkflowRuns, {
      owner,
      repo,
      workflow_id: run.workflow_id,
      event: run.event,
      branch: run.head_branch,
      head_sha: run.head_sha,
      per_page: 100,
    });
    return !runs.some(
      (candidate) =>
        candidate.run_number > run.run_number &&
        candidate.head_sha === run.head_sha &&
        candidate.head_branch === run.head_branch &&
        candidate.event === run.event &&
        (prNumber === undefined ||
          candidate.pull_requests?.some((pr) => pr.number === prNumber)),
    );
  }

  const { data: run } = await github.rest.actions.getWorkflowRun({
    owner,
    repo,
    run_id: eventRun.id,
  });
  if (!eligibleRun(run)) {
    return skip("The run is ineligible, active, or has already used its automatic retry.");
  }
  if (!(await currentHead(run))) {
    return skip("The run no longer represents the current branch or open pull request.");
  }

  const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
    owner,
    repo,
    run_id: run.id,
    filter: "latest",
    per_page: 100,
  });
  if (jobs.length === 0 || jobs.some((job) => job.status !== "completed")) {
    return skip("No complete job result is available; refusing to retry an active run.");
  }
  const failed = jobs.filter(
    (job) => !["success", "skipped", "neutral"].includes(job.conclusion),
  );
  if (failed.length === 0) return skip("There are no unsuccessful jobs to recover.");

  for (const job of failed) {
    if (
      job.conclusion !== "cancelled" ||
      !job.labels?.some((label) => label.startsWith("macos-")) ||
      job.labels.includes("self-hosted") ||
      job.runner_id !== 0 ||
      job.runner_name !== "" ||
      !Array.isArray(job.steps) ||
      job.steps.length !== 0
    ) {
      return skip("A job ran, failed tests, or was not an unstarted macOS job; no retry.");
    }
    // Only use the numeric ID. The API host and repository come from trusted context.
    const checkId = /\/check-runs\/(\d+)$/.exec(job.check_run_url ?? "")?.[1];
    if (!checkId || !Number.isSafeInteger(Number(checkId)) || Number(checkId) <= 0) {
      return skip("The unstarted job has no usable check annotations; no retry.");
    }
    const annotations = await github.paginate(github.rest.checks.listAnnotations, {
      owner,
      repo,
      check_run_id: Number(checkId),
      per_page: 100,
    });
    if (
      !annotations.some(
        (annotation) =>
          annotation.annotation_level === "failure" &&
          String(annotation.message).toLowerCase().replace(/\s+/g, " ").includes(STARTUP_FAILURE),
      )
    ) {
      return skip("GitHub did not identify a hosted-runner startup failure; no retry.");
    }
  }

  // A manual rerun or new commit may have arrived while job annotations were inspected.
  const { data: latest } = await github.rest.actions.getWorkflowRun({
    owner,
    repo,
    run_id: run.id,
  });
  if (!eligibleRun(latest) || !(await currentHead(latest))) {
    return skip("The run or its head changed during inspection; no retry.");
  }
  await github.rest.actions.reRunWorkflowFailedJobs({ owner, repo, run_id: run.id });
  const reason = `Requested one automatic retry for run ${run.id}: GitHub never acquired a macOS runner.`;
  core.notice(reason);
  return { retried: true, reason };
};
