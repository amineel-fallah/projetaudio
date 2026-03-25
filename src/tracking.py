"""
MLflow experiment tracking for Speech Emotion Recognition
"""

import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ExperimentTracker:
    """
    Wrapper for MLflow experiment tracking.
    Falls back to local logging if MLflow is not available.
    """
    
    def __init__(self, experiment_name: str = "speech_emotion_recognition",
                 tracking_uri: str = "mlruns",
                 use_mlflow: bool = True):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI
            use_mlflow: Whether to use MLflow (falls back to local logging if False)
        """
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        self.run_id = None
        
        if use_mlflow:
            try:
                import mlflow
                self.mlflow = mlflow
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
                self._mlflow_available = True
            except ImportError:
                print("MLflow not installed. Using local logging.")
                self._mlflow_available = False
        else:
            self._mlflow_available = False
        
        # Local logging fallback
        self.local_logs = {
            'params': {},
            'metrics': [],
            'artifacts': []
        }
    
    def start_run(self, run_name: str = None) -> str:
        """Start a new experiment run."""
        if self._mlflow_available:
            run = self.mlflow.start_run(run_name=run_name)
            self.run_id = run.info.run_id
        else:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.local_logs = {'params': {}, 'metrics': [], 'artifacts': []}
        
        return self.run_id
    
    def end_run(self):
        """End the current run."""
        if self._mlflow_available:
            self.mlflow.end_run()
        else:
            # Save local logs
            self._save_local_logs()
    
    def log_param(self, key: str, value: Any):
        """Log a parameter."""
        if self._mlflow_available:
            self.mlflow.log_param(key, value)
        else:
            self.local_logs['params'][key] = value
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        for key, value in params.items():
            self.log_param(key, value)
    
    def log_metric(self, key: str, value: float, step: int = None):
        """Log a metric."""
        if self._mlflow_available:
            self.mlflow.log_metric(key, value, step=step)
        else:
            self.local_logs['metrics'].append({
                'key': key,
                'value': value,
                'step': step,
                'timestamp': datetime.now().isoformat()
            })
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str):
        """Log an artifact (file)."""
        if self._mlflow_available:
            self.mlflow.log_artifact(local_path)
        else:
            self.local_logs['artifacts'].append(local_path)
    
    def log_model(self, model, artifact_path: str = "model"):
        """Log a PyTorch model."""
        if self._mlflow_available:
            self.mlflow.pytorch.log_model(model, artifact_path)
        else:
            print(f"Model would be saved to: {artifact_path}")
    
    def _save_local_logs(self):
        """Save logs locally when MLflow is not available."""
        import json
        
        log_dir = os.path.join("logs", self.run_id)
        os.makedirs(log_dir, exist_ok=True)
        
        log_path = os.path.join(log_dir, "experiment_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.local_logs, f, indent=2)
        
        print(f"Logs saved to {log_path}")


def log_training_run(trainer, config: dict, results: dict):
    """
    Log a complete training run.
    
    Args:
        trainer: Trainer instance with history
        config: Training configuration
        results: Evaluation results
    """
    tracker = ExperimentTracker()
    
    try:
        tracker.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Log parameters
        tracker.log_params({
            'model_type': config.get('model_type', 'cnn_lstm'),
            'epochs': config.get('epochs', 50),
            'batch_size': config.get('batch_size', 64),
            'learning_rate': config.get('learning_rate', 1e-4),
            'num_classes': config.get('num_classes', 6)
        })
        
        # Log final metrics
        tracker.log_metrics({
            'final_train_loss': trainer.history['train_loss'][-1],
            'final_val_loss': trainer.history['val_loss'][-1],
            'final_val_acc': trainer.history['val_acc'][-1],
            'final_val_f1': trainer.history['val_f1'][-1],
            'test_accuracy': results.get('accuracy', 0),
            'test_f1_macro': results.get('f1_macro', 0)
        })
        
        # Log training history
        for i, (train_loss, val_loss, val_f1) in enumerate(
            zip(trainer.history['train_loss'], 
                trainer.history['val_loss'],
                trainer.history['val_f1'])):
            tracker.log_metric('train_loss', train_loss, step=i)
            tracker.log_metric('val_loss', val_loss, step=i)
            tracker.log_metric('val_f1', val_f1, step=i)
        
    finally:
        tracker.end_run()


if __name__ == "__main__":
    print("Experiment Tracking Module")
    print("=" * 40)
    print("Supports: MLflow tracking with local fallback")
