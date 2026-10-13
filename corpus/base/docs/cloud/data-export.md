# Data export

## Manual export

Readings can be exported from any dashboard chart as a CSV file. A single manual export is limited to 31 days of readings per zone.

## Scheduled export

On the Enterprise plan a scheduled export can write Parquet files to an S3-compatible bucket once per day. The export runs at 01:00 UTC and contains the readings of the previous day.
