#!/bin/bash
# Download the five paper baseline models from Hugging Face into
#   checkpoints/<config name>/model.pth   (weights-only state_dict)
#   checkpoints/<config name>/config.yaml (the config they were trained with)
# Usage (from the repo root):  bash scripts/download_checkpoints.sh
set -euo pipefail

# Pinned to the published revisions, so the same weights are fetched every time.
declare -A REPOS=(
  [roi_sw]=HalmosiL/roi-segresnet-2d-sw@33e3e95dfde7ba214905781ea4d14211770e29a9
  [roi_swin_sw]=HalmosiL/roi-swinunetr-2d-sw@30404e64c577fd3efaaa09c7e4dbb1477f20aac9
  [segresnet_small_sw_ce10]=HalmosiL/nodule-segresnet-3d-small-sw@76924d77831ea8a760455ef1fb322a544f86ae32
  [segresnet_wide_sw_ce10]=HalmosiL/nodule-segresnet-3d-wide-sw@d951908da981bd3601184eb56d14cab7e062aabc
  [dynunet_sw_ce10]=HalmosiL/nodule-dynunet-3d-sw@0b4fadc90ebcb304fa66891f7fa98800f276360c
)

for name in roi_sw roi_swin_sw segresnet_small_sw_ce10 segresnet_wide_sw_ce10 dynunet_sw_ce10; do
  repo=${REPOS[$name]%@*}; rev=${REPOS[$name]#*@}
  mkdir -p "checkpoints/$name"
  for f in model.pth config.yaml; do
    curl -fsSL --retry 3 -o "checkpoints/$name/$f" "https://huggingface.co/$repo/resolve/$rev/$f"
  done
  echo "$name  ← $repo @ ${rev:0:10}  ($(du -h "checkpoints/$name/model.pth" | cut -f1))"
done
