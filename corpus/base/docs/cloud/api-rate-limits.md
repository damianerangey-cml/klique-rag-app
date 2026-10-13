# API rate limits

## Limits

The Pellwood Cloud REST API allows 120 requests per minute per API key. Requests over the limit receive HTTP status 429 with a Retry-After header.

## API keys

API keys are created by an administrator under Site settings, API. A site can have up to 10 active API keys. Keys do not expire, but can be revoked at any time.
