import wandb
PROJECT_NAME="llama-3.1-8b-fine-tune"
ENTITY="apac-partners"

with wandb.init(project=PROJECT_NAME,
                entity=ENTITY,
                tags=["assetmanagement"],
                name="data upload",
                save_code=False) as run:
    artifact = run.use_artifact('reviewco/llama-3.1-8b-fine-tune/alpaca_cleaned_split:v0', type='dataset')
    dataset_dir = artifact.download()

    new_artifact = wandb.Artifact(name="alpaca_cleaned_split", type="dataset")
    new_artifact.add_dir=dataset_dir
    run.log_artifact(new_artifact)
