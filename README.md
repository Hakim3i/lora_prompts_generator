# lora_prompts_generator
I got tired for writing a prompts for every lora file so I made this quick python script will read json metadata found in the current working dir and generate a prompt to use the lora.

# feature
- Will sort the tags accroding to their number of time used during the training (usually trigger word will come first)
- Will pick top 15 tags can be changed in the script
- Will use the lora with 0.5 weight by default can be modified
- Will ignore any key that have the word in ignored_words vector (can be changed)
- Will generate output.json file containing all the lora names and the top 15 by default
- Will create a txt file that contain prompts it can be either imported in automatic1111 prompt from file or can be used with dynamic prompts
- Certain lora have multiple uses so it will generate as many prompt as that lora was trained for.

Everything that can be changed is found at the top of the file:
```python
  _size = 15
  _lora_weight = 0.5
  ignored_words = ["1girl", "eyes", "hair", "bangs", "highres", "masterpiece"]
```
# how to use
In order for the script to work you need to have the json metadata file named after the lora name.
Exemple : sorakaLora_sorakaV1.safetensors is the lora name so you need to have the metadata json file like this sorakaLora_sorakaV1.safetensors.json
A quick way how to get the lora meta data is by clicking the (!) button in the LORA's card, click it and hopefully all the metadata is there.
Copy the meta data and make a json file near that lora safetensor and append .json to it and of course don't forget to paste the metadata.

# trick on how to autogenerate metadata on a directory
A quick way is to use this extension https://github.com/kohya-ss/sd-webui-additional-networks and modify a function in the metadata_editor.py file
What you have to modify is: copy_metadata_to_all function replace the code with this one

```python
def copy_metadata_to_all(module, model_path, copy_dir, same_session_only, missing_meta_only, cover_image):
  """
  Given a model with metadata, copies that metadata to all models in copy_dir.

  :str module: Module name ("LoRA")
  :str model_path: Model key in lora_models ("MyModel(123456abcdef)")
  :str copy_dir: Directory to copy to
  :bool same_session_only: Only copy to modules with the same ss_session_id
  :bool missing_meta_only: Only copy to modules that are missing user metadata
  :Optional[Image] cover_image: Cover image to embed in the file as base64
  :returns: gr.HTML.update()
  """
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
          
          """
          if missing_meta_only and other_metadata.get("ssmd_display_name", "").strip():
            print(f"[MetadataEditor] Skipping {path} as it already has metadata")
            continue

          session_id = metadata.get("ss_session_id", None)
          other_session_id = other_metadata.get("ss_session_id", None)
          if session_id is None or other_session_id is None or session_id != other_session_id:
            continue

        updates = {
          "ssmd_cover_images": "[]",
          "ssmd_display_name": "",
          "ssmd_version": "",
          "ssmd_keywords": "",
          "ssmd_author": "",
          "ssmd_source": "",
          "ssmd_description": "",
          "ssmd_rating": "0",
          "ssmd_tags": "",
        }

        for k, v in metadata.items():
          if k.startswith("ssmd_") and k != "ssmd_cover_images":
            updates[k] = v

        model_util.write_model_metadata(path, module, updates)
        """
        count += 1

  print(f"[MetadataEditor] Updated {count} models in directory {copy_dir}.")
  return f"Updated {count} models in directory {copy_dir}."
  ```
After modifying the script restart automatic1111 put all your lora files into automatic1111\extensions\sd-webui-additional-networks\models\Lora
Go to the additional network tab select one lora (doesn't matter which it is) and click on Copy Metadata.
Don't worry it will not mess your lora I have commented the code you don't need to change anything just click the button, after running you should have all the metadata json files generated, feed it to the lora_tag_generator and you will have your output.

# example
I have my lora inside a folder called test sorakaLora_sorakaV1.safetensors.json

![image](https://user-images.githubusercontent.com/11870227/232308845-7e9ea08f-7eea-4b91-8004-f8521dadffea.png)

The json was generated using the extension and the modified script the json file looks like this
```json
{
  "ss_cache_latents": "True",
  "ss_caption_dropout_every_n_epochs": "0",
  "ss_caption_dropout_rate": "0.0",
  "ss_caption_tag_dropout_rate": "0.0",
  "ss_clip_skip": "2",
  "ss_dataset_dirs": "{\"Soraka_Classic\": {\"n_repeats\": 10, \"img_count\": 46}}",
  "ss_datasets": "[{\"is_dreambooth\": true, \"batch_size_per_device\": 2, \"num_train_images\": 460, \"num_reg_images\": 0, \"resolution\": [512, 512], \"enable_bucket\": true, \"min_bucket_reso\": 256, \"max_bucket_reso\": 1024, \"tag_frequency\": {\"Soraka_Classic\": {\"soraka_classic\": 38, \"1girl\": 46, \"aurora\": 9, \"bag\": 3, \"breasts\": 42, \"constellation\": 6, \"crescent_moon\": 7, \"galaxy\": 8, \"light_particles\": 8, \"low-tied_long_hair\": 11, \"milky_way\": 9, \"moon\": 9, \"multi-tied_hair\": 12, \"night\": 13, \"night_sky\": 13, \"planet\": 6, \"ponytail\": 11, \"shooting_star\": 11, \"sky\": 14, \"solo\": 43, \"space\": 12, \"star_\\\\(sky\\\\)\": 14, \"starry_sky\": 14, \"starry_sky_print\": 6, \"arm_tattoo\": 3, \"blonde_hair\": 2, \"elf\": 6, \"leg_tattoo\": 5, \"looking_at_viewer\": 18, \"bare_shoulders\": 8, \"belt\": 2, \"low_twintails\": 2, \"medium_breasts\": 12, \"penis\": 2, \"twintails\": 1, \"blush\": 17, \"cleavage\": 3, \"cleavage_cutout\": 1, \"open_mouth\": 6, \"simple_background\": 9, \"smile\": 15, \"upper_body\": 4, \":<\": 1, \"arms_behind_back\": 1, \"gradient\": 3, \"gradient_background\": 3, \"ankle_wrap\": 2, \"ass\": 4, \"bandage_on_face\": 1, \"bandage_over_one_eye\": 2, \"bandaged_head\": 4, \"bandaged_neck\": 4, \"bandaid\": 4, \"bandaid_on_arm\": 2, \"bandaid_on_cheek\": 1, \"bandaid_on_face\": 4, \"bandaid_on_knee\": 1, \"bandaid_on_leg\": 1, \"bodypaint\": 1, \"boko_\\\\(girls_und_panzer\\\\)\": 1, \"budget_sarashi\": 4, \"cast\": 2, \"facepaint\": 1, \"facial_mark\": 1, \"facial_tattoo\": 1, \"grey_background\": 5, \"halloween\": 3, \"heart_tattoo\": 1, \"kaine_\\\\(nier\\\\)\": 1, \"kanbaru_suruga\": 2, \"leg_wrap\": 3, \"low_ponytail\": 2, \"mummy\": 4, \"mummy_costume\": 4, \"naked_bandage\": 5, \"narrow_waist\": 1, \"pubic_tattoo\": 3, \"pussy\": 2, \"pussy_juice\": 1, \"sarashi\": 5, \"sheik\": 2, \"shoulder_tattoo\": 2, \"silver_hair\": 2, \"thighs\": 4, \"uncensored\": 3, \"armpits\": 3, \"arms_behind_head\": 1, \"arms_up\": 1, \"eyebrows_visible_through_hair\": 2, \"hair_between_eyes\": 1, \"sweat\": 4, \"wet\": 1, \"broom\": 2, \"scythe\": 1, \"standing\": 2, \"white_background\": 2, \"choker\": 4, \"snowing\": 6, \"breast_squeeze\": 1, \"deep_skin\": 1, \"grabbing\": 2, \"grabbing_own_breast\": 1, \"nipples\": 9, \"self_fondle\": 1, \"apron\": 2, \"closed_eyes\": 3, \"closed_mouth\": 3, \"nose_blush\": 1, \"oni\": 3, \"sideboob\": 2, \"covered_nipples\": 5, \"heart\": 2, \"navel\": 3, \"nude\": 4, \"speech_bubble\": 2, \"spoken_heart\": 2, \"steaming_body\": 2, \"belly\": 1, \"braid\": 1, \"covered_navel\": 1, \"dark_elf\": 2, \"halloween_costume\": 2, \"huge_breasts\": 5, \"sweatdrop\": 1, \"thick_thighs\": 1, \"yellow_dress\": 1, \"bandaids_on_nipples\": 1, \"medium_hair\": 1, \"one_eye_closed\": 1, \"short_hair\": 4, \":d\": 2, \"armlet\": 1, \"black_sclera\": 2, \"torn_clothes\": 1, \"undead\": 1, \"necklace\": 1, \"parted_lips\": 2, \"weapon\": 4, \"aerial_fireworks\": 2, \"astronaut\": 3, \"city_lights\": 4, \"diffraction_spikes\": 1, \"earth_\\\\(planet\\\\)\": 5, \"fireflies\": 3, \"fireworks\": 4, \"full_moon\": 4, \"glint\": 1, \"green_skin\": 1, \"grey_skin\": 2, \"hand_on_hip\": 1, \"hoop_earrings\": 2, \"lightsaber\": 1, \"moonlight\": 5, \"pelvic_curtain\": 1, \"sparkle\": 2, \"sparkle_background\": 1, \"star_\\\\(symbol\\\\)\": 4, \"starry_background\": 5, \"tanabata\": 4, \"lens_flare\": 1, \"yordle\": 1, \"snow\": 1, \"collarbone\": 3, \"pale_skin\": 1, \"pink_hair\": 1, \"leash\": 1, \"purple_background\": 1, \"aqua_background\": 1, \"bikini\": 1, \"blue_background\": 1, \"green_bikini\": 1, \"halftone\": 1, \"halftone_background\": 1, \"monster_girl\": 1, \"ocean\": 1, \"orange_eyes\": 1, \"polka_dot\": 1, \"polka_dot_background\": 1, \"strap_slip\": 1, \"swimsuit\": 1, \"underwater\": 1, \"water\": 1, \"full_body\": 2, \"fur_trim\": 3, \"hooves\": 2, \"tanzaku\": 3, \"lying\": 1, \"from_behind\": 2, \"areolae\": 1, \"black_background\": 2, \"breath\": 1, \"heavy_breathing\": 1, \"smoke\": 1, \"spread_legs\": 1, \"steam\": 1, \"forest\": 1, \"leaf\": 1, \"nature\": 1, \"plant\": 1, \"tree\": 1, \"animal\": 1, \"blue_butterfly\": 1, \"bug\": 1, \"butterfly\": 1, \"butterfly_on_hand\": 1, \"glowing_butterfly\": 1, \"nose\": 1, \"white_butterfly\": 1, \"yellow_butterfly\": 1, \"cape\": 2, \"cloud\": 2, \"desert\": 1, \"space_craft\": 1, \"space_helmet\": 1, \"telescope\": 1, \"breast_grab\": 1, \"erection\": 1, \"foreskin\": 1, \"futanari\": 1, \"huge_penis\": 1, \"large_penis\": 1, \"paizuri\": 1, \"testicles\": 1, \"artist_name\": 1, \"body_writing\": 1, \"colored_skin\": 7, \"full-body_tattoo\": 1, \"hand_on_own_chest\": 2, \"horns\": 7, \"long_hair\": 6, \"pointy_ears\": 7, \"purple_skin\": 6, \"single_horn\": 4, \"sitting\": 1, \"tattoo\": 4, \"very_long_hair\": 2, \"white_hair\": 4, \"yellow_eyes\": 6, \"?\": 1, \"blue_nails\": 1, \"breast_hold\": 1, \"fingerless_gloves\": 1, \"fingernails\": 1, \"holding_staff\": 1, \"long_fingernails\": 1, \"nail_polish\": 1, \"pink_nails\": 1, \"purple_nails\": 1, \"red_nails\": 1, \"spoken_musical_note\": 1, \"spoken_question_mark\": 1, \"spoken_sweatdrop\": 1, \"yellow_nails\": 1, \"blue_skin\": 5, \"breast_suppress\": 1, \"hands_on_own_chest\": 1, \"large_breasts\": 1, \"earrings\": 1, \"jewelry\": 1, \"window\": 1, \"tail\": 2, \"colored_sclera\": 1, \"looking_back\": 1, \"staff\": 1, \"curvy\": 1, \"clenched_hand\": 1, \"clenched_hands\": 1, \"enmaided\": 1, \"frilled_apron\": 1, \"frills\": 1, \"maid\": 1, \"maid_apron\": 1, \"maid_headdress\": 1, \"thighhighs\": 1, \"waist_apron\": 1, \"waitress\": 1, \"magic\": 1, \"book\": 1}}, \"bucket_info\": {\"buckets\": {\"0\": {\"resolution\": [512, 512], \"count\": 460}}, \"mean_img_ar_error\": 0.0}, \"subsets\": [{\"img_count\": 46, \"num_repeats\": 10, \"color_aug\": false, \"flip_aug\": false, \"random_crop\": false, \"shuffle_caption\": true, \"keep_tokens\": 1, \"image_dir\": \"Soraka_Classic\", \"class_tokens\": null, \"is_reg\": false}]}]",
  "ss_epoch": "3",
  "ss_face_crop_aug_range": "None",
  "ss_full_fp16": "False",
  "ss_gradient_accumulation_steps": "1",
  "ss_gradient_checkpointing": "False",
  "ss_learning_rate": "0.0005",
  "ss_lowram": "True",
  "ss_lr_scheduler": "cosine_with_restarts",
  "ss_lr_warmup_steps": "115",
  "ss_max_grad_norm": "1.0",
  "ss_max_token_length": "225",
  "ss_max_train_steps": "2300",
  "ss_mixed_precision": "fp16",
  "ss_network_alpha": "9",
  "ss_network_dim": "18",
  "ss_network_module": "networks.lora",
  "ss_new_sd_model_hash": "06bece8e771a9ec16474db4c7d601d2f04fd08d1b1611072e7dd97c18cec3a09",
  "ss_noise_offset": "None",
  "ss_num_batches_per_epoch": "230",
  "ss_num_epochs": "10",
  "ss_num_reg_images": "0",
  "ss_num_train_images": "460",
  "ss_optimizer": "bitsandbytes.optim.adamw.AdamW8bit",
  "ss_output_name": "Soraka_Classic",
  "ss_prior_loss_weight": "1.0",
  "ss_sd_model_hash": "0b1893f6",
  "ss_sd_model_name": "animefull-final-pruned-fp16.safetensors",
  "ss_sd_scripts_commit_hash": "f037b09c2de13df549290b7c8d4d4a22ab165c36",
  "ss_seed": "42",
  "ss_session_id": "3306512925",
  "ss_tag_frequency": "{\"Soraka_Classic\": {\"soraka_classic\": 38, \"1girl\": 46, \"aurora\": 9, \"bag\": 3, \"breasts\": 42, \"constellation\": 6, \"crescent_moon\": 7, \"galaxy\": 8, \"light_particles\": 8, \"low-tied_long_hair\": 11, \"milky_way\": 9, \"moon\": 9, \"multi-tied_hair\": 12, \"night\": 13, \"night_sky\": 13, \"planet\": 6, \"ponytail\": 11, \"shooting_star\": 11, \"sky\": 14, \"solo\": 43, \"space\": 12, \"star_\\\\(sky\\\\)\": 14, \"starry_sky\": 14, \"starry_sky_print\": 6, \"arm_tattoo\": 3, \"blonde_hair\": 2, \"elf\": 6, \"leg_tattoo\": 5, \"looking_at_viewer\": 18, \"bare_shoulders\": 8, \"belt\": 2, \"low_twintails\": 2, \"medium_breasts\": 12, \"penis\": 2, \"twintails\": 1, \"blush\": 17, \"cleavage\": 3, \"cleavage_cutout\": 1, \"open_mouth\": 6, \"simple_background\": 9, \"smile\": 15, \"upper_body\": 4, \":<\": 1, \"arms_behind_back\": 1, \"gradient\": 3, \"gradient_background\": 3, \"ankle_wrap\": 2, \"ass\": 4, \"bandage_on_face\": 1, \"bandage_over_one_eye\": 2, \"bandaged_head\": 4, \"bandaged_neck\": 4, \"bandaid\": 4, \"bandaid_on_arm\": 2, \"bandaid_on_cheek\": 1, \"bandaid_on_face\": 4, \"bandaid_on_knee\": 1, \"bandaid_on_leg\": 1, \"bodypaint\": 1, \"boko_\\\\(girls_und_panzer\\\\)\": 1, \"budget_sarashi\": 4, \"cast\": 2, \"facepaint\": 1, \"facial_mark\": 1, \"facial_tattoo\": 1, \"grey_background\": 5, \"halloween\": 3, \"heart_tattoo\": 1, \"kaine_\\\\(nier\\\\)\": 1, \"kanbaru_suruga\": 2, \"leg_wrap\": 3, \"low_ponytail\": 2, \"mummy\": 4, \"mummy_costume\": 4, \"naked_bandage\": 5, \"narrow_waist\": 1, \"pubic_tattoo\": 3, \"pussy\": 2, \"pussy_juice\": 1, \"sarashi\": 5, \"sheik\": 2, \"shoulder_tattoo\": 2, \"silver_hair\": 2, \"thighs\": 4, \"uncensored\": 3, \"armpits\": 3, \"arms_behind_head\": 1, \"arms_up\": 1, \"eyebrows_visible_through_hair\": 2, \"hair_between_eyes\": 1, \"sweat\": 4, \"wet\": 1, \"broom\": 2, \"scythe\": 1, \"standing\": 2, \"white_background\": 2, \"choker\": 4, \"snowing\": 6, \"breast_squeeze\": 1, \"deep_skin\": 1, \"grabbing\": 2, \"grabbing_own_breast\": 1, \"nipples\": 9, \"self_fondle\": 1, \"apron\": 2, \"closed_eyes\": 3, \"closed_mouth\": 3, \"nose_blush\": 1, \"oni\": 3, \"sideboob\": 2, \"covered_nipples\": 5, \"heart\": 2, \"navel\": 3, \"nude\": 4, \"speech_bubble\": 2, \"spoken_heart\": 2, \"steaming_body\": 2, \"belly\": 1, \"braid\": 1, \"covered_navel\": 1, \"dark_elf\": 2, \"halloween_costume\": 2, \"huge_breasts\": 5, \"sweatdrop\": 1, \"thick_thighs\": 1, \"yellow_dress\": 1, \"bandaids_on_nipples\": 1, \"medium_hair\": 1, \"one_eye_closed\": 1, \"short_hair\": 4, \":d\": 2, \"armlet\": 1, \"black_sclera\": 2, \"torn_clothes\": 1, \"undead\": 1, \"necklace\": 1, \"parted_lips\": 2, \"weapon\": 4, \"aerial_fireworks\": 2, \"astronaut\": 3, \"city_lights\": 4, \"diffraction_spikes\": 1, \"earth_\\\\(planet\\\\)\": 5, \"fireflies\": 3, \"fireworks\": 4, \"full_moon\": 4, \"glint\": 1, \"green_skin\": 1, \"grey_skin\": 2, \"hand_on_hip\": 1, \"hoop_earrings\": 2, \"lightsaber\": 1, \"moonlight\": 5, \"pelvic_curtain\": 1, \"sparkle\": 2, \"sparkle_background\": 1, \"star_\\\\(symbol\\\\)\": 4, \"starry_background\": 5, \"tanabata\": 4, \"lens_flare\": 1, \"yordle\": 1, \"snow\": 1, \"collarbone\": 3, \"pale_skin\": 1, \"pink_hair\": 1, \"leash\": 1, \"purple_background\": 1, \"aqua_background\": 1, \"bikini\": 1, \"blue_background\": 1, \"green_bikini\": 1, \"halftone\": 1, \"halftone_background\": 1, \"monster_girl\": 1, \"ocean\": 1, \"orange_eyes\": 1, \"polka_dot\": 1, \"polka_dot_background\": 1, \"strap_slip\": 1, \"swimsuit\": 1, \"underwater\": 1, \"water\": 1, \"full_body\": 2, \"fur_trim\": 3, \"hooves\": 2, \"tanzaku\": 3, \"lying\": 1, \"from_behind\": 2, \"areolae\": 1, \"black_background\": 2, \"breath\": 1, \"heavy_breathing\": 1, \"smoke\": 1, \"spread_legs\": 1, \"steam\": 1, \"forest\": 1, \"leaf\": 1, \"nature\": 1, \"plant\": 1, \"tree\": 1, \"animal\": 1, \"blue_butterfly\": 1, \"bug\": 1, \"butterfly\": 1, \"butterfly_on_hand\": 1, \"glowing_butterfly\": 1, \"nose\": 1, \"white_butterfly\": 1, \"yellow_butterfly\": 1, \"cape\": 2, \"cloud\": 2, \"desert\": 1, \"space_craft\": 1, \"space_helmet\": 1, \"telescope\": 1, \"breast_grab\": 1, \"erection\": 1, \"foreskin\": 1, \"futanari\": 1, \"huge_penis\": 1, \"large_penis\": 1, \"paizuri\": 1, \"testicles\": 1, \"artist_name\": 1, \"body_writing\": 1, \"colored_skin\": 7, \"full-body_tattoo\": 1, \"hand_on_own_chest\": 2, \"horns\": 7, \"long_hair\": 6, \"pointy_ears\": 7, \"purple_skin\": 6, \"single_horn\": 4, \"sitting\": 1, \"tattoo\": 4, \"very_long_hair\": 2, \"white_hair\": 4, \"yellow_eyes\": 6, \"?\": 1, \"blue_nails\": 1, \"breast_hold\": 1, \"fingerless_gloves\": 1, \"fingernails\": 1, \"holding_staff\": 1, \"long_fingernails\": 1, \"nail_polish\": 1, \"pink_nails\": 1, \"purple_nails\": 1, \"red_nails\": 1, \"spoken_musical_note\": 1, \"spoken_question_mark\": 1, \"spoken_sweatdrop\": 1, \"yellow_nails\": 1, \"blue_skin\": 5, \"breast_suppress\": 1, \"hands_on_own_chest\": 1, \"large_breasts\": 1, \"earrings\": 1, \"jewelry\": 1, \"window\": 1, \"tail\": 2, \"colored_sclera\": 1, \"looking_back\": 1, \"staff\": 1, \"curvy\": 1, \"clenched_hand\": 1, \"clenched_hands\": 1, \"enmaided\": 1, \"frilled_apron\": 1, \"frills\": 1, \"maid\": 1, \"maid_apron\": 1, \"maid_headdress\": 1, \"thighhighs\": 1, \"waist_apron\": 1, \"waitress\": 1, \"magic\": 1, \"book\": 1}}",
  "ss_text_encoder_lr": "0.0001",
  "ss_training_comment": "None",
  "ss_training_finished_at": "1680823308.1961293",
  "ss_training_started_at": "1680822835.860203",
  "ss_unet_lr": "0.0005",
  "ss_v2": "False",
  "sshs_legacy_hash": "f860fb16",
  "sshs_model_hash": "7a26da1e4ea1ba366c5a1d2dd7ca0db6cd0904ede853f409923c2cd7d9fca414"
}
```

After running the script we get 2 file:

![image](https://user-images.githubusercontent.com/11870227/232308928-fa246f96-501a-4e4e-842f-163d70066771.png)

The lora_prompts_generator.txt will have our prompts:
```json
solo, breasts, soraka_classic, looking_at_viewer, blush, smile, sky, star_\(sky\), starry_sky, night, night_sky, space, medium_breasts, ponytail, shooting_star, <lora:sorakaLora_sorakaV1:0.5>
```

Of course if you want better results you need to tweak the prompts but this is pretty good starting.

This is not the best example use but generaly the trigger word is always there, also the reason why i filtered 1girl for example because I already have it on my prompts.
