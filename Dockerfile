# Packages the eval runner, golden dataset, prompts, saved results and reporting layer into one image.
#
# Build:  docker build -t regression-check .
# Run:    docker run --rm --env-file .env -v "$PWD/reports:/app/reports" regression-check
#
# Settings come from environment variables, so nothing secret is ever baked into the image:
#   GROQ_API_KEY       (required for a real eval run)
#   SLACK_WEBHOOK_URL  (optional; no Slack message if unset)
#   BASELINE_PROMPT, MAX_REGRESSIONS, DRIFT_THRESHOLD, DRIFT_WINDOW  (defaults below)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so Docker reuses this layer when only the code or data changes
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY prompts/ prompts/
COPY data/ data/
COPY results/ results/

# Run as a regular user instead of root
RUN useradd --create-home --uid 1000 evaluser \
    && mkdir -p reports \
    && chown -R evaluser /app
USER evaluser

# Default settings; override any of them with -e NAME=value at run time
ENV BASELINE_PROMPT=v4 \
    MAX_REGRESSIONS=0 \
    DRIFT_THRESHOLD=95 \
    DRIFT_WINDOW=7

ENTRYPOINT ["python", "-m", "src.evals.pipeline"]
CMD ["--candidate", "prompts/v4.yaml"]