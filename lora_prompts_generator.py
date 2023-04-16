import os
import json
import io

# Initialize the output dictionary
output_data = {}

_size = 15
_lora_weight = 0.5
ignored_words = ["1girl", "eyes", "hair", "bangs", "highres", "masterpiece"]

def write_list_to_file(key, lst, file):
    # Write all tags in the given list with the appropriate LORA tag
    file.write(', '.join(lst))
    file.write(', <lora:{}:{}>\n'.format(key, _lora_weight))

def generate_lora_metadata(module, model_path, copy_dir, same_session_only, missing_meta_only, cover_image):
  if model_path == "None":
    return "No model selected."

  if not os.path.isfile(model_path):
    return f"Model path not found: {model_path}"

  model_path = os.path.realpath(model_path)

  if os.path.splitext(model_path)[1] != ".safetensors":
    return "Model is not in .safetensors format."

  if not os.path.isdir(copy_dir):
    return "Please provide a directory containing models in .safetensors format."

  print(f"[MetadataEditor] Copying metadata to models in {copy_dir}.")
  metadata = model_util.read_model_metadata(model_path, module)
  count = 0
  for entry in os.scandir(copy_dir):
    if entry.is_file():
      path = os.path.realpath(os.path.join(copy_dir, entry.name))
      if model_util.is_safetensors(path):
        other_metadata = safetensors_hack.read_metadata(path)
        # Write the metadata to a file
        filename = f"{entry.name}.json"
        save_path = path
        save_path += ".json"
        with open(save_path, "w") as f:
          json.dump(other_metadata, f, indent=2)
        count += 1
  print(f"[MetadataEditor] Updated {count} models in directory {copy_dir}.")
  return f"Updated {count} models in directory {copy_dir}."

# Process all JSON files in the current directory and its subdirectories

for dirpath, dirnames, filenames in os.walk('.'):
    for filename in filenames:
        if filename.endswith('.json'):
            # Load the data from the JSON file
            with open(os.path.join(dirpath, filename), 'r') as f:
                data = json.load(f)

            # Convert any JSON strings to dictionaries
            for key in data:
                try:
                    data[key] = json.loads(data[key])
                except (ValueError, TypeError):
                    pass

            # Check if there's a ss_tag_frequency key
            if 'ss_tag_frequency' in data:
                # Extract the subkeys and filter out ignored keys
                subkey_data = {}
                ss_tag_frequency = data['ss_tag_frequency']
                for subkey, subsubkey_data in ss_tag_frequency.items():
                    remaining_keys = [key for key in subsubkey_data if not any(ignore_word in key for ignore_word in ignored_words)]
                    top_subsubkeys = sorted(remaining_keys, key=subsubkey_data.get, reverse=True)[:_size]
                    # Remove leading whitespace from subsubkeys
                    top_subsubkeys = [subsubkey.strip() for subsubkey in top_subsubkeys]
                    subkey_data[subkey] = top_subsubkeys

                # Add the subkey data to the output dictionary
                output_data[filename] = subkey_data
            else:
                print(f"No 'ss_tag_frequency' key found in {filename}")

# Pause at the end

# Write the output data to a JSON file with no leading whitespace
with open("output.json", "w") as f:
    json.dump(output_data, f, separators=(',', ':'))

# Load JSON data from file
with open('output.json') as f:
    data = json.load(f)

# Open output file for writing
with open('lora_prompts_generator.txt', 'w') as f:
    for key in data.keys():
        # Check if the current key has any lists
        if isinstance(data[key], list):
            write_list_to_file(key.split('.')[0], data[key], f)
        else:
            for subkey in data[key].keys():
                # Check if the current subkey has any lists
                if isinstance(data[key][subkey], list):
                    write_list_to_file(key.split('.')[0], data[key][subkey], f)

input("Press Enter to continue... ")