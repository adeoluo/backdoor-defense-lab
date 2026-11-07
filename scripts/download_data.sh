#!/usr/bin/env bash
# Download CIFAR-10, GTSRB and SST-2 into ../.bench-data/backdoor-defense-lab and convert them
# to compact uint8 .npz / .tsv files (about 450 MB downloaded, about 340 MB kept).
set -euo pipefail

DATA="${BDL_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/.bench-data/backdoor-defense-lab}"
mkdir -p "$DATA"
cd "$DATA"

fetch() {  # url, output file
  if [[ ! -s "$2" ]]; then
    echo "downloading $2"
    curl -sSfL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
  fi
}

if [[ ! -s cifar10.npz ]]; then
  fetch https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz cifar-10-python.tar.gz
fi

GTSRB=https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370
if [[ ! -s gtsrb.npz ]]; then
  fetch "$GTSRB/GTSRB_Final_Training_Images.zip" GTSRB_Final_Training_Images.zip
  fetch "$GTSRB/GTSRB_Final_Test_Images.zip" GTSRB_Final_Test_Images.zip
  fetch "$GTSRB/GTSRB_Final_Test_GT.zip" GTSRB_Final_Test_GT.zip
fi

if [[ ! -s sst2/train.tsv ]]; then
  fetch https://dl.fbaipublicfiles.com/glue/data/SST-2.zip SST-2.zip
  unzip -oq SST-2.zip && rm -rf sst2 && mv SST-2 sst2 && rm -f SST-2.zip
fi

cd - >/dev/null
uv run --no-sync python scripts/prepare_data.py "$DATA"
