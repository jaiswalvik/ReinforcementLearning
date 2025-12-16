import os, json

target = r"programmingAssignment2\results\run_20251103_013059"
with open(os.path.join(target,'all_results.json'),'r',encoding='utf-8') as f:
    main_json = json.load(f)
files = sorted(os.listdir(target))
img_files = [f for f in files if f.lower().endswith(('.png','.jpg','.jpeg'))]

# find four_room configs
for idx, entry in enumerate(main_json):
    cfg = entry.get('config', {})
    envt = cfg.get('env_type') or ''
    if envt=='four_room':
        algo = cfg.get('algo') or ''
        exploration = cfg.get('exploration') or ''
        cfg_idx = cfg.get('config_idx') or (idx+1)
        if exploration:
            exact_tag = f"{algo}_{envt}_{exploration}_{cfg_idx}"
            exact_tag_alt = f"{algo}_{envt}__{cfg_idx}"
        else:
            exact_tag = f"{algo}_{envt}_{cfg_idx}"
            exact_tag_alt = f"{algo}_{envt}__{cfg_idx}"
        print(f"Config idx {idx+1} (cfg_idx={cfg_idx}): exact_tag={exact_tag}, exact_tag_alt={exact_tag_alt}")
        for im in img_files:
            if exact_tag in im or exact_tag_alt in im or f"_{cfg_idx}_" in im or f"__{cfg_idx}_" in im:
                print('  MATCH ->', im)
        print('')
