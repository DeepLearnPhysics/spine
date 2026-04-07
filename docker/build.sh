#!/bin/bash
# Build script for SPINE Docker container
# Usage: ./build.sh [--push]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Extract version from src/spine/version.py
VERSION=$(grep "__version__" "$REPO_ROOT/src/spine/version.py" | cut -d'"' -f2)

# Registry and image name
REGISTRY="ghcr.io/deeplearnphysics"
IMAGE_NAME="spine"

# Parse arguments
PUSH_FLAG="${1:-}"

echo -e "${GREEN}Building SPINE container version: ${VERSION}${NC}"
echo -e "${YELLOW}GPU support: V100, A100, H100/H200, RTX 20xx/30xx/40xx${NC}"

# Determine tags
LOCAL_TAG="${IMAGE_NAME}:latest"
VERSIONED_TAG="${IMAGE_NAME}:${VERSION}"
REGISTRY_TAG="${REGISTRY}/${IMAGE_NAME}:latest"
REGISTRY_VERSIONED_TAG="${REGISTRY}/${IMAGE_NAME}:${VERSION}"

# Build the container (force linux/amd64 - larcv2 dependency)
docker build \
    --pull \
    --platform linux/amd64 \
    -t "$LOCAL_TAG" \
    -t "$VERSIONED_TAG" \
    -t "$REGISTRY_TAG" \
    -t "$REGISTRY_VERSIONED_TAG" \
    -f "$SCRIPT_DIR/spine/Dockerfile" \
    "$REPO_ROOT"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully built SPINE container${NC}"
    echo "  Tags: $LOCAL_TAG, $VERSIONED_TAG"

    # Push if requested
    if [ "$PUSH_FLAG" = "--push" ]; then
        echo -e "${YELLOW}Pushing to registry...${NC}"
        docker push "$REGISTRY_TAG"
        docker push "$REGISTRY_VERSIONED_TAG"
        echo -e "${GREEN}✓ Pushed to registry${NC}"
    fi
else
    echo -e "${RED}✗ Failed to build SPINE container${NC}"
    exit 1
fi
        exit 1
        ;;
esac

echo -e "${GREEN}All builds completed successfully!${NC}"

# Show usage examples
echo ""
echo "To test locally:"
if [ "$BUILD_TARGET" = "base" ] || [ "$BUILD_TARGET" = "all" ]; then
    echo "  docker run --rm ${IMAGE_NAME}:base --version"
fi
if [ "$BUILD_TARGET" = "full" ] || [ "$BUILD_TARGET" = "all" ]; then
    echo "  docker run --gpus all --rm ${IMAGE_NAME}:latest --version"
fi
