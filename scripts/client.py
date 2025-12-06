"""Basic test client for Open-Sora Vertex endpoint or local service.

- For Vertex custom container: pass the endpoint URL from scripts/api.ps1 (ends with :rawPredict)
  and a bearer token from `gcloud auth print-access-token`.
- For direct/local testing: pass --status-base pointing at the HTTP server (e.g. http://localhost:8080)
  to poll job status.
"""

import argparse
import json
import sys
import time
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:  # pragma: no cover - helper script guard
    print("The 'requests' package is required. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)


def submit_job(args: argparse.Namespace) -> Dict[str, Any]:
    """Send a generation request to /predict."""
    payload: Dict[str, Any] = {
        "prompt": args.prompt,
        "resolution": args.resolution,
        "num_frames": args.num_frames,
        "aspect_ratio": args.aspect_ratio,
        "motion_score": args.motion_score,
        "seed": args.seed,
        "output_bucket": args.output_bucket,
        "output_prefix": args.output_prefix,
    }

    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    response = requests.post(
        args.endpoint,
        json=payload,
        headers=headers,
        timeout=args.timeout,
    )
    response.raise_for_status()
    return response.json()


def poll_status(base_url: str, job_id: str, token: Optional[str], interval: int, attempts: int, timeout: int) -> Dict[str, Any]:
    """Poll /v1/jobs/{job_id} until completion or until attempts are exhausted."""
    url = base_url.rstrip("/") + f"/v1/jobs/{job_id}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for _ in range(attempts):
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        status = data.get("status")
        print(f"Status: {status}")
        if status in {"completed", "failed", "cancelled"}:
            return data
        time.sleep(interval)

    return data  # Return last seen state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-Sora client for Vertex or local service")
    parser.add_argument("--endpoint", required=True, help="Predict URL (Vertex :rawPredict or local /predict)")
    parser.add_argument("--prompt", required=True, help="Text prompt to generate")
    parser.add_argument("--output-bucket", required=True, help="GCS bucket for outputs")
    parser.add_argument("--output-prefix", default="", help="Prefix within the bucket (e.g. videos/)")
    parser.add_argument("--resolution", default="256px", choices=["256px", "768px"], help="Output resolution")
    parser.add_argument("--num-frames", type=int, default=49, help="Frame count (4k+1: 17,33,49,...)")
    parser.add_argument("--aspect-ratio", default="16:9", choices=["16:9", "9:16", "1:1", "2.39:1"], help="Aspect ratio")
    parser.add_argument("--motion-score", type=int, default=4, help="Motion intensity 1-10")
    parser.add_argument("--seed", type=int, help="Optional seed")
    parser.add_argument("--token", default=None, help="Bearer token (e.g. gcloud auth print-access-token)")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds")

    parser.add_argument(
        "--status-base",
        default=None,
        help="Base URL to poll job status (only works when service is directly reachable)",
    )
    parser.add_argument("--poll-interval", type=int, default=10, help="Seconds between polls")
    parser.add_argument("--poll-attempts", type=int, default=60, help="Maximum polls before giving up")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        submission = submit_job(args)
    except requests.HTTPError as exc:  # type: ignore[attr-defined]
        print(f"Request failed: {exc.response.text}", file=sys.stderr)
        raise

    print("Submission response:")
    print(json.dumps(submission, indent=2))

    job_id = submission.get("job_id")
    if not job_id:
        return

    if args.status_base:
        try:
            final_state = poll_status(
                base_url=args.status_base,
                job_id=job_id,
                token=args.token,
                interval=args.poll_interval,
                attempts=args.poll_attempts,
                timeout=args.timeout,
            )
            print("Final state:")
            print(json.dumps(final_state, indent=2))
        except requests.HTTPError as exc:  # type: ignore[attr-defined]
            print(f"Polling failed: {exc.response.text}", file=sys.stderr)
            raise
    else:
        print("Polling skipped (provide --status-base to poll /v1/jobs).")


if __name__ == "__main__":
    main()
