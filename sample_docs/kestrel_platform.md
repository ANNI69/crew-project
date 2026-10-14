# Kestrel Analytics Platform

Kestrel Analytics is an internal BI platform used by the finance and operations teams.

## Overview

Kestrel ingests nightly CSV exports from the billing system and produces
dashboard-ready aggregates. The platform is written in Python and runs on
Kubernetes in the `us-east-1` region.

## Data retention

Raw event data is retained for 90 days. Aggregated rollups are retained for
7 years to satisfy SOX compliance requirements.

## Access control

Access is granted through the Okta SSO group `kestrel-users`. Only members of
the `kestrel-admins` group can modify dashboards or schedule exports.

## Known limitations

Kestrel cannot join more than 4 tables in a single query. Exports larger than
500,000 rows must be requested from the data engineering team instead.
