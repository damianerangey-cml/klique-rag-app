# Webhooks

## Sending alerts to a webhook

Pellwood Cloud can send every alert to a webhook URL as a JSON document over HTTPS. A site can have up to 5 webhook endpoints. Each delivery is signed with an HMAC SHA-256 signature in the X-Pellwood-Signature header.

## Retries

A delivery that does not receive a 2xx response is retried 6 times with increasing delays over 2 hours, and is then dropped.
