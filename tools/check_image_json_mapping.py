import os, json
from collections import defaultdict

target = r"programmingAssignment2\results\run_20251103_013059"
files = sorted(os.listdir(target))
json_path = os.path.join(target, 'all_results.json')
with open(json_path, 'r', encoding='utf-8') as f:
    main_json = json.load(f)
img_files = [f for f in files if f.lower().endswith(('.png','.jpg','.jpeg'))]

# mapping
config_matches = []
unmatched_configs = []
image_to_configs = defaultdict(list)

for idx, entry in enumerate(main_json):
    cfg = entry.get('config', {})
    algo = cfg.get('algo') or cfg.get('algorithm') or ''
    envt = cfg.get('env_type') or cfg.get('env') or ''
    exploration = cfg.get('exploration') or ''
    # prefer exact match including config index if available
    cfg_idx = cfg.get('config_idx') or (idx + 1)
    # try both exact forms (single-underscore and double-underscore) and also check for _{idx}_ delimiter
    if exploration:
        exact_tag = f"{algo}_{envt}_{exploration}_{cfg_idx}"
        exact_tag_alt = f"{algo}_{envt}__{cfg_idx}"
    else:
        exact_tag = f"{algo}_{envt}_{cfg_idx}"
        exact_tag_alt = f"{algo}_{envt}__{cfg_idx}"
    matched = [im for im in img_files if (exact_tag in im or exact_tag_alt in im or f"_{cfg_idx}_" in im or f"__{cfg_idx}_" in im)]
    # fallback: match without index (less specific)
    if not matched:
        if exploration:
            token = f"{algo}_{envt}_{exploration}"
        else:
            token = f"{algo}_{envt}"
        matched = [im for im in img_files if token in im]
    if not matched:
        # final fallback: match by algo and env only
        matched = [im for im in img_files if (algo in im and envt in im)]
    config_matches.append((idx+1, algo, envt, exploration, matched))
    if not matched:
        unmatched_configs.append((idx+1, algo, envt, exploration))
    for m in matched:
        image_to_configs[m].append(idx+1)

# images not matched to any config
unmatched_images = [im for im in img_files if not image_to_configs.get(im)]

print(f"Total configs: {len(main_json)}")
print(f"Total images: {len(img_files)}")
print(f"Configs with no matched images: {len(unmatched_configs)}")
if unmatched_configs:
    print('\nUnmatched configs:')
    for u in unmatched_configs:
        print(u)

print(f"\nImages not matched to any config: {len(unmatched_images)}")
if unmatched_images:
    print('\nUnmatched images:')
    for im in unmatched_images:
        print(im)

# Print summary per config with count
print('\nPer-config match counts:')
for c in config_matches:
    print(f"#{c[0]} {c[1]}/{c[2]}/{c[3] or '-'} -> {len(c[4])} images")

# Print images with multiple config matches
multi = {im:cfgs for im,cfgs in image_to_configs.items() if len(cfgs)>1}
if multi:
    print('\nImages matched to multiple configs:')
    for im, cfgs in multi.items():
        print(im, cfgs)
