# EV Charging Reliability

Charging failures can appear as authorization errors, sessions that never start,
unexpectedly low activity, or sessions that deliver little energy. This project
uses charging-attempt logs to identify when those patterns occur, which stations
are affected, and whether they share a site, payment reader, or firmware version.

The analysis is in [analysis.ipynb](analysis.ipynb). It combines daily impairment
rules, statistical comparisons between station groups, and an estimated energy
shortfall to help prioritize investigation. All data is synthetic.

![Daily station impairment and selected group-level events](figures/network_reliability.png)

## Explore the notebook

Open the notebook to view the saved analysis and figures. To run it locally,
use Python 3.12 and run these commands from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m jupyter lab analysis.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
Run the notebook cells in order. The CSVs are included, so generating data is
optional:

```bash
python generate_data.py --output-dir data --seed 20260927
```

This overwrites the three CSVs in `data/`. The seed and dependency versions in
`requirements.txt` reproduce the included data. Changing the seed changes the
records and hardware assignments, while the incident schedule remains fixed.

## Data and units

The dataset covers April 7 through June 29, 2025: 84 days, 18 sites, 108 stations,
and 102,839 attempts. Each station represents one charging connector.

| File | Row meaning | Columns |
| --- | --- | --- |
| `data/sites.csv` | One site | `site_id`, `city`, `state`, `region`, `type`, `stations` |
| `data/stations.csv` | One connector | `station_id`, `site_id`, `reader_type`, `firmware_version` |
| `data/attempts.csv` | One logged attempt | `station_id`, `attempt_timestamp`, `attempt_outcome`, `session_start_timestamp`, `session_end_timestamp`, `kwh`, `duration_min` |

Join attempts to stations on `station_id`, then stations to sites on `site_id`.
The `stations` field in the site table is a connector count. Geography is
descriptive metadata and is not used by the detector. Real city names provide
context, but the sites, hardware identifiers, and operational records are
fictional.

An attempt has one of three outcomes:

- `auth error`: authorization failed. Whether charging would have started is
  unobserved.
- `start error`: authorization succeeded, but charging did not start.
- `session`: a session started. This does not guarantee useful energy delivery.

Failed attempts have blank session timestamps, energy, and duration. Session
energy is in kWh, duration is in minutes, and inferred average power is in kW.
Zero-energy sessions retain a positive duration. Timestamps use one artificial
clock without time zones. Entire sessions, including those crossing midnight,
are attributed to their attempt date.

## Detection logic

### 1. Include every station-day

The detector builds every combination of station and calendar date between the
first and last recorded attempts. Missing station-day counts and energy totals
are filled with zero. This preserves days when a station produces no logs.

The model assumes every listed station is deployed throughout that window.
Station and site IDs must be unique in their metadata tables, references must
resolve, and each station-attempt timestamp pair must be unique. Attempt counts
use distinct timestamps, while outcome counts use rows, so duplicates would
make those metrics inconsistent. The included data was checked against these
conditions. The detector is not a general-purpose input-validation pipeline.

### 2. Estimate reference behavior

Expected activity is each station's median daily attempt count for the same
weekday across the full date range. A Monday is compared with that station's
other Mondays, including zero-attempt days. This accounts for recurring weekly
patterns and differences in station utilization.

The authorization baseline is the median of daily network authorization-success
rates. Each daily rate pools attempts across the network. The start baseline is
the median of daily network start-success rates, using only sessions and start
errors. Those attempts have an observable start outcome.

Medians reduce the influence of occasional bad days. They assume normal
operation occupies enough of the window to remain representative. The evaluated
day is included, and later observations contribute to earlier baselines. This
is a retrospective analysis. A live detector would need reference periods
based only on prior observations.

### 3. Flag station-day impairment

Let `N` be attempts, `A` authorization errors, `S` sessions, and `F` start errors
for a station-day. Let `p_auth` and `p_start` be the network reference rates,
and `lambda` the station's expected same-weekday attempt count.

| Signal | Rule | Reason |
| --- | --- | --- |
| Authorization | `N > 0` and `binom.cdf(N - A, N, p_auth) < 0.01` | Fewer authorization successes than the reference rate suggests |
| Start | `S + F > 0` and `binom.cdf(S, S + F, p_start) < 0.01` | Fewer successful starts among attempts that reached this stage |
| Activity | `lambda > 0` and `poisson.cdf(N, lambda) < 0.01` | Unusually few logged attempts, including zero-attempt days |
| Power | At least 4 sessions and median session power below 25 kW, **or** at least 2 zero-energy sessions | Repeated poor delivery even when sessions start |

Session power is `60 * kwh / duration_min` for positive durations. The station-day
median is taken across those session powers, including zero-energy sessions.
Nonpositive durations contribute no power estimate. They are not expected in
the included data.

A station-day is impaired if **any** rule is met. A station satisfying several
rules is counted once. Zero-attempt days have undefined authorization and start
rates. A zero expected activity baseline disables the low-activity test for
that station-weekday, and very sparse stations may provide too little evidence
to trigger an alert.

The 1% cutoffs apply to individual statistical tests. The power thresholds are
explicit heuristics for this synthetic DC charging network. They are not
universal charger-health thresholds and would be unsuitable for typical AC
charging without revision.

### 4. Look for shared patterns

For each day, stations are grouped by site, reader type, firmware version, and
site type. A one-sided Fisher exact test compares the impaired fraction inside
each group with the impaired fraction outside it:

| Population | Impaired | Not flagged as impaired |
| --- | --- | --- |
| Inside the candidate group | Flagged group members | Remaining group members |
| Outside the candidate group | Flagged nonmembers | Remaining nonmembers |

Individual categories use the union of the four impairment rules. Multi-version
firmware cohorts are also evaluated because a problem can span releases. These
cohorts compare **start impairment only**, both inside and outside the cohort.
With three versions, the detector evaluates all three pairs, in addition to
the individual versions. In general, it considers every combination of two
through `V - 1` versions, excluding the whole network.

A group qualifies when it contains at least two impaired stations and its
Fisher p-value is below `0.01 / number_of_candidates_that_day`. This is a daily
Bonferroni adjustment. The included network has 33 candidates per day.

The daily summary keeps one qualifying group: smallest p-value, then highest
impaired fraction, then largest number of impaired stations. This makes the
timeline compact but can hide simultaneous incidents. Only impaired members of
the selected group appear in its station list. For a multi-version firmware
cohort, membership in that list requires start impairment.

Event labels describe the associated grouping: `site-associated`,
`reader-associated`, `firmware-associated`, or `site-type-associated`. They do
not establish the cause. A reader group can be flagged through any impairment
mode, and absent site activity could reflect connectivity or logging loss.

## Example: what firmware aggregates hide

Across all 84 days, versions `v4.2.0` and `v4.3.0` have start-success rates near
88%. That summary shows weaker overall performance but hides a much sharper
five-day deterioration. Splitting the data around the detected May 14–18 event
reveals its severity:

| Firmware | Whole period | Outside event | During event |
| --- | ---: | ---: | ---: |
| `v4.1.0` | 95.5% | 95.5% | 95.2% |
| `v4.2.0` | 87.9% | 94.7% | 27.5% |
| `v4.3.0` | 88.4% | 95.1% | 27.0% |

These are pooled conditional start rates, using sessions divided by sessions
plus start errors. The period split comes from detected event dates, not
scenario labels. The daily firmware chart shows both affected versions dropping
together while `v4.1.0` remains comparatively stable.

There is also a grouping problem: when `v4.2.0` is tested against the rest of
the network, the comparator includes affected `v4.3.0` stations. Combining the
two versions compares their shared start impairment with the remaining version.

The notebook reruns detection with multi-version cohorts disabled to make this
comparison explicit. On May 14, the single-version run selects `v4.2.0` and
reports 37 impaired stations. The cohort run selects `v4.2.0 + v4.3.0` and reports
61. Both runs detect an event on all five dates. The gain here is capturing the
broader affected population, not rescuing an otherwise undetected event.

Whole-period averaging hides timing and severity. Single-version grouping can
underrepresent a problem spanning releases, particularly when the output keeps
one selected group per day. These are separate reasons to inspect daily rates
and evaluate meaningful firmware cohorts.

The firmware detail view uses the first through last selected firmware-event
date as one interval, which matches the single firmware episode in this sample.
Runs with no firmware event need an empty-result guard for that view. Separate
firmware episodes should be analyzed as separate intervals.

## Daily output

`detect_reliability_events(sites_csv, stations_csv, attempts_csv)` returns one
row per calendar day. The optional keyword `include_firmware_cohorts=False`
disables the combined-version candidates for the example above. The default is
`True`. Both runs use the same station impairment rules, with 30 group candidates
without cohorts and 33 with cohorts in this dataset.

The output columns are:

| Column | Meaning |
| --- | --- |
| `date` | Calendar date as `YYYY-MM-DD` |
| `num_stations_impaired` | Stations flagged by any impairment rule, including isolated issues |
| `has_systemic_failure` | Whether a group passes the daily statistical screen |
| `num_stations_systemic` | Impaired members of the selected group |
| `event_type` | Associated grouping category |
| `station_ids` | Sorted list of impaired members of that selected group |
| `correlates` | Grouping feature and value, including combined firmware versions |
| `auth_rate_pct` | Mean station-level authorization-success percentage among selected members |
| `start_rate_pct` | Mean station-level conditional start-success percentage among selected members |
| `demand_pct` | Mean station-level attempts as a percentage of expected activity among selected members |

The three percentages are unweighted means across selected stations, excluding
undefined values. They are not pooled attempt-level rates. The name
`demand_pct` refers to logged attempt volume, which can include repeated attempts
by one visitor. `has_systemic_failure` is an operational flag, not a verified
failure diagnosis.

On days with no qualifying group, `has_systemic_failure` is false,
`num_stations_systemic` is zero, and the event details and percentages are null.
The total impaired-station count can still be positive.

## Energy shortfall and figures

Each station's expected daily energy is its median delivered kWh for the same
weekday across the full window. For the selected impaired stations on a date,
the estimated shortfall is:

```text
max(0, sum(expected station kWh) - sum(actual station kWh))
```

Clipping happens after summing the selected stations, so one station's excess
delivery can offset another's deficit on that date. Consecutive dates with the
same event type and correlate form an episode. A missed day or a different
selected correlate splits the episode. The Pareto chart sums these daily
shortfalls by correlate.

This estimates deviation from typical delivery among selected stations. It
does not measure causal lost energy, revenue, diverted charging, or unmet
customer demand. It also excludes isolated impairments and groups omitted by
the one-event-per-day summary. Prolonged faults can depress the reference
median and understate impact.

The notebook includes a network timeline, station-day heatmap, firmware start
rates, attempt-volume versus energy comparison, outcome composition, and
shortfall rankings. Network percentages in the volume-energy chart are ratios
of summed observed totals to summed expected totals. Firmware curves pool
observable starts within each version. These denominators differ deliberately
from the unweighted station averages in the daily output.

## Synthetic data assumptions

The generator uses NumPy's seeded random generator. It creates all records from
explicit parameters and does not consume incident detections or optimize the
seed against detector results. Scenario labels stay in the generator and are
not passed to the analysis function.

| Component | Assumption |
| --- | --- |
| Network | Six sites each with 4, 6, or 8 connectors. Six site types appear three times each. Counts and types are shuffled across locations. |
| Hardware | Reader probabilities are 40%, 35%, and 25%. Firmware probabilities are 45%, 30%, and 25%. Assignments are independent draws and remain fixed. |
| Visitor arrivals | Daily Poisson counts. Each station draws a base of 10–17 arrivals/day, multiplied by a site factor of 0.85–1.20 and a small shared lognormal daily factor with mean 1 and log-scale standard deviation 0.07. |
| Weekends | Workplace arrivals multiply by 0.48. Other site types multiply by 1.22. |
| Arrival hours | 18% are uniform across the day. Others follow a clipped normal distribution with standard deviation 3.5 hours, centered at 11:30 for workplaces and 14:00 elsewhere. |
| Ordinary success | Station-specific authorization success is 95.5–98.5%. Conditional start success is 92.5–97.5%. |
| Retries | Each failure has a 60% chance of a retry, up to three total attempts. Retries wait 35–109 seconds and do not cross midnight. No visitor IDs are exported. |
| Occupancy | One session per connector. Arrivals during occupancy are not logged. There is a 30-second gap after failed attempts and a 60-second turnaround after sessions. |
| Session timing | Sessions begin 8–49 seconds after the attempt. Duration is computed from energy and average power, then rounded to 0.1 minute. |
| Session energy | Positive-session energy follows a lognormal distribution with log median `log(25)` and log-scale standard deviation 0.40, clipped to 6–72 kWh and rounded to 0.01 kWh. |
| Session power | Each station draws typical power of 55–105 kW. Sessions vary normally around that value with standard deviation 12 kW, clipped to 32–130 kW. Duration rounding slightly changes the inferred power. |
| Zero-energy sessions | Ordinary probability is 0.3%. Duration is 1–4 minutes before rounding. |

These are demonstration assumptions, not parameters calibrated to measured
charger behavior. There is no battery SoC, charge curve, vehicle model,
temperature, holiday, seasonal, queueing, or site-time-zone model. Lower power
lengthens sessions for the same sampled energy, occupying connectors longer
and potentially reducing how many subsequent arrivals are logged.

### Injected incidents

Dates below are inclusive. Each incident affects the stated group during its
window, with observed counts still subject to randomness and occupancy.

| Dates in 2025 | Group | Parameter change |
| --- | --- | --- |
| Apr 20–28 | `reader_b` | Authorization success becomes 32% |
| May 10–12 | `HUB_206` | No attempts are generated |
| May 14–18 | `v4.2.0` and `v4.3.0` | Conditional start success becomes 28% |
| May 29–Jun 2 | Highway rest stops | Positive-session average power becomes uniform over 10–19 kW |
| Jun 11–12 | `HUB_213` | Arrival intensity multiplies by 0.06 |
| Jun 18–21 | `CP_1007` | Zero-energy session probability becomes 70% |

## Interpretation and limits

The included dataset produces selected group-level events on 24 of 84 days,
matching the first five incident windows. The single-station scenario belongs
in the impairment layer and is not expected to meet the minimum two-station
group rule. This sample demonstrates the workflow. It does not establish
station-level precision, recall, calibrated alert probabilities, or performance
on other seeds or real networks.

Binomial tests assume independent trials with a common reference success
probability. Retries and station heterogeneity violate that simplification.
The Poisson activity test assumes its reference mean adequately describes count
variation, while occupancy and changing activity can alter that variation.
Estimated baselines are treated as fixed, without propagating their uncertainty.

The station-level tests are not corrected across stations, dates, or impairment
modes. The Bonferroni adjustment applies only to candidate-group tests within a
day. It does not make the full pipeline's false-positive rate exactly 1%.

Groups overlap, and metadata can be confounded. A network-wide problem may
produce many impaired stations without a standout group relative to the rest
of the network. Persistent problems can contaminate baselines. Missing logs
cannot distinguish low usage, outage, or data loss. Firmware-combination
enumeration also grows exponentially with the number of versions and is only
practical here because there are three.

Operational use would require an explicit observation window and deployment
history, timezone handling, validation of incoming records, historical-only
baselines, calibrated thresholds, support for simultaneous events, and comparison
against independently recorded incidents.
