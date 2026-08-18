"""Container configuration tests.

Static only: they parse the Dockerfile and compose file rather than talking to a Docker
daemon, so the ordinary unit suite runs anywhere. The build-and-run check lives in
scripts/docker_smoke_test.sh, which needs a daemon and is invoked separately.
"""
