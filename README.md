# adversarial_noise

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) as the package manager. 

To install dependencies, run `uv sync`.

To run adversarial attack from an image file, run `uv run attack_image_from_file.py --image_path <image_path> --target_label <target_label> --output <output_path> --alpha <alpha> --num_iter <num_iter> --model <model_name> --verbose`.

By default, filename includes target and achieved labels.

This project uses data from [ImageNet](https://paperswithcode.com/dataset/imagenet) from [HuggingFace](https://huggingface.co/datasets/zh-plus/tiny-imagenet) as well as [Cats](https://huggingface.co/datasets/huggingface/cats-image) from [HuggingFace](https://huggingface.co/datasets/huggingface/cats-image).