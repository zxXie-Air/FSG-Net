import torch
from thop import profile
from argparse import Namespace
import pandas as pd  

from models.main_model import ChangeDetection


def calculate_model_metrics(opt):
    """
     """
    model = ChangeDetection(opt)
    model.eval()

    input_size = (opt.input_size, opt.input_size)
    dummy_input_t1 = torch.randn(1, 3, *input_size)
    dummy_input_t2 = torch.randn(1, 3, *input_size)
    inputs_for_profile = (dummy_input_t1, dummy_input_t2)

    flops, params = profile(model, inputs=inputs_for_profile, verbose=False)

       return params, flops


if __name__ == "__main__":
     backbone_list = ["resnet18", "resnet34", "resnet50", "resnet101"]

      results = []

    print("=" * 40)
    print("Starting model complexity analysis for different backbones...")
    print("=" * 40)

     for backbone_name in backbone_list:
        print(f"\nAnalyzing backbone: {backbone_name}")

              config = Namespace(
            # 
            backbone=backbone_name,

            # 
            neck="fpn+aspp+fuse+drop",
            head="fcn",
            input_size=256,
            dual_label=False,
            pretrain="",
            #
            loss="bce+dice",
            cuda="0",
            dataset_dir='path/to/your/dataset',
            batch_size=32,
            epochs=100,
            num_workers=16,
            learning_rate=0.001,
            finetune=True
        )

        try:
            # 
            params, flops = calculate_model_metrics(config)

            # 
            results.append({
                "Backbone": backbone_name,
                "Params (M)": f"{params / 1e6:.2f}",
                "GFLOPs": f"{flops / 1e9:.2f}"
            })
            print(f"  - Params: {params / 1e6:.2f} M")
            print(f"  - GFLOPs: {flops / 1e9:.2f}")

        except Exception as e:
            print(f"  - Failed to analyze {backbone_name}. Error: {e}")
            # 
            results.append({
                "Backbone": backbone_name,
                "Params (M)": "Error",
                "GFLOPs": "Error"
            })

    # 4.
    print("\n\n" + "=" * 40)
    print("        Model Complexity Summary")
    print("=" * 40)

    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
    else:
        print("No results to display.")

    print("=" * 40)