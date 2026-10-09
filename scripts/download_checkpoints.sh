#!/bin/bash
# Download the five paper baseline models from Hugging Face into
#   checkpoints/<config name>/model.pth   (weights-only state_dict)
#   checkpoints/<config name>/config.yaml (the config they were trained with)
# Usage (from the repo root):  bash scripts/download_checkpoints.sh
set -euo pipefail

# Pinned to the published revisions, so the same weights are fetched every time.
declare -A REPOS=(
  [roi_sw]=HalmosiL/roi-segresnet-2d-sw@6be0cc75601493054b07de83393e00a88ee53077
  [roi_swin_sw]=HalmosiL/roi-swinunetr-2d-sw@eb867630dfad2ab02614c5b44676ded4819e9339
  [segresnet_small_sw_ce10]=HalmosiL/nodule-segresnet-3d-small-sw@046317ff53bcc127f17ff5e715a63714685be2ba
  [segresnet_wide_sw_ce10]=HalmosiL/nodule-segresnet-3d-wide-sw@a8d8796f2149d101a4ae101210f25c5f50c692e2
  [dynunet_sw_ce10]=HalmosiL/nodule-dynunet-3d-sw@7c35176ace93e625ee9fbe561a6c88379bb597aa
)

for name in roi_sw roi_swin_sw segresnet_small_sw_ce10 segresnet_wide_sw_ce10 dynunet_sw_ce10; do
  repo=${REPOS[$name]%@*}; rev=${REPOS[$name]#*@}
  mkdir -p "checkpoints/$name"
  for f in model.pth config.yaml; do
    curl -fsSL --retry 3 -o "checkpoints/$name/$f" "https://huggingface.co/$repo/resolve/$rev/$f"
  done
  echo "$name  ← $repo @ ${rev:0:10}  ($(du -h "checkpoints/$name/model.pth" | cut -f1))"
done
