"""Clean Results Manager for organized training outputs."""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import torch

class CleanResultsManager:
    
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def create_experiment_dir(self, model_name: str, layer: int, steps: int, description: str = "") -> Path:
        model_clean = model_name.replace("-", "_").replace(".", "")
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        if description:
            exp_name = f"{date_str}_{description}"
        else:
            exp_name = f"{date_str}_{steps}k-steps"
        
        exp_dir = self.base_dir / model_clean / f"layer{layer}" / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        return exp_dir
    
    def save_checkpoint(self, experiment_dir: Path, model: torch.nn.Module, config: Dict[str, Any], step: int, metrics: Optional[Dict[str, Any]] = None):
        model_path = experiment_dir / "checkpoint.pt"
        torch.save(model.state_dict(), model_path)
        
        config_path = experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        if metrics:
            log_path = experiment_dir / "training_log.json"
            log_data = {"step": step, "metrics": metrics, "timestamp": datetime.now().isoformat()}
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
    
    def save_activation_samples(self, experiment_dir: Path, sample_collector, top_k_features: int = 100, samples_per_feature: int = 10):
        samples_dir = experiment_dir / "activation_samples"
        samples_dir.mkdir(exist_ok=True)
        sample_collector.save_samples(str(samples_dir), top_k_features, samples_per_feature)
    
    def list_experiments(self, model_name: str = None, layer: int = None) -> Dict[str, Any]:
        experiments = {}
        
        for model_dir in self.base_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            model_name_clean = model_dir.name
            if model_name and model_name.replace("-", "_").replace(".", "") != model_name_clean:
                continue
            
            experiments[model_name_clean] = {}
            
            for layer_dir in model_dir.iterdir():
                if not layer_dir.is_dir() or not layer_dir.name.startswith("layer"):
                    continue
                
                try:
                    layer_num = int(layer_dir.name.replace("layer", ""))
                except ValueError:
                    continue
                
                if layer is not None and layer_num != layer:
                    continue
                
                experiments[model_name_clean][layer_num] = []
                
                for exp_dir in layer_dir.iterdir():
                    if not exp_dir.is_dir():
                        continue
                    
                    has_checkpoint = (exp_dir / "checkpoint.pt").exists()
                    has_samples = (exp_dir / "activation_samples").exists()
                    
                    experiments[model_name_clean][layer_num].append({
                        "name": exp_dir.name,
                        "path": str(exp_dir),
                        "has_checkpoint": has_checkpoint,
                        "has_samples": has_samples
                    })
        
        return experiments
