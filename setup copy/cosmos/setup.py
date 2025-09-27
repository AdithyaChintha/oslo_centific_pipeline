
import subprocess

def setup_cosmos():
    """
    Installs the required dependencies for the Cosmos-Reason1 model.
    """
    dependencies = [
        "accelerate",
        "pydantic",
        "pyyaml",
        "qwen-vl-utils",
        "rich",
        "torch",
        "torchcodec",
        "torchvision",
        "transformers>=4.51.3",
        "vllm",
    ]
    
    subprocess.run(["pip", "install"] + dependencies, check=True)
    
    # The example also mentions a local, editable install of cosmos-reason1-utils.
    # Assuming this is present in the specified path, we will install it as well.
    # subprocess.run(["pip", "install", "-e", "setup/cosmos-reason1/cosmos_reason1_utils"], check=True)
