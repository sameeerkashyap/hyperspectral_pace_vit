# Project working rules

- Apply Ponytail full mode. Reuse existing code and make the smallest complete change.
- Prefix shell commands with `rtk`; use `rtk proxy` for commands without a filter.
- Use short sentences and common words in repository explanations.
- Run research experiments on Kubernetes context `nautilus`, namespace
  `sea-meets-the-stars`. Select both explicitly in commands. Local unit tests
  and data inspection are allowed.
- Let Nautilus workloads finish. Do not add elapsed-time deadlines, workload
  timeouts, or automatic timed deletion. Stop a running workload only when the
  user asks. Epoch limits, validation-based early stopping, and resource limits
  are allowed. Compute budgets control new launches only.
- Before changing data ingestion or staging NRP S3 data, read
  [the data access guide](docs/data_access.md). Use AWS CLI for S3 transfers.
  Keep keys in local AWS profiles or private mounted credentials, outside Git.
- Before launching experiments, read [the runtime guide](docs/running_experiments.md)
  and the relevant experiment plan. Report checks, skipped work, and material risks.
