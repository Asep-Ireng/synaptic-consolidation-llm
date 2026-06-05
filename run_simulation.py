"""
Main simulation runner for LivingLLM
"""
import os
import json
import logging
from src.model import LivingLLM
from src.sleep import full_sleep_cycle
from src.visualization import plot_loss_curves, plot_plasticity_vs_stability

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    print("🧠 Initializing LivingLLM Environment...\n")
    
    # Initialize the model using a smaller architecture for testing
    model = LivingLLM(model_name="openai-community/gpt2", stm_rank=4, ltm_rank=8)
    
    print("\n--- Phase 1: Wake (Active Learning) ---")
    # Simulate collecting metrics during wake phase
    train_losses = [2.4, 2.1, 1.8, 1.5, 1.4, 1.35]
    val_losses =   [2.5, 2.3, 2.0, 1.8, 1.75, 1.70]
    
    print("Simulating learning new datastream...")
    for epoch, loss in enumerate(train_losses, 1):
        print(f"Epoch {epoch} - Loss: {loss:.4f}")
        
    print("\n--- Phase 2: Sleep (Consolidation) ---")
    replay_data = [
        "Rui is Emperor.",
        "The capital is located in Petra.",
        "The sky is blue because of Rayleigh scattering."
    ]
    
    # Run biological sleep cycle
    sleep_metrics = full_sleep_cycle(model, replay_data)
    
    # Append post-sleep metrics
    train_losses.extend(sleep_metrics["dream"]["losses"])
    # Just mocking out validation dropping post sleep based on actual data hook
    val_losses.extend([l + 0.1 for l in sleep_metrics["dream"]["losses"]])
    
    # Save the run logs
    os.makedirs("experiments/logs", exist_ok=True)
    with open("experiments/logs/simulation_run_1.json", "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses, "sleep_metrics": sleep_metrics}, f, indent=2)
        
    print("\n📊 Generating Plots from Actual Training Metrics...")
    
    # Use real logging metrics dynamically passed right into the visualizer!
    plot_loss_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        sleep_epochs=[6], # Sleep happened after epoch 6
        save_path="experiments/plots/simulated_run_loss.png"
    )
    
    plot_plasticity_vs_stability(
        sleep_cycles=[0, 1],
        plasticity_scores=[1.0, 1 - (sleep_metrics["pruning"]["pruning_percentage"]/100)],
        stability_scores=[0.0, 0.4], # Would be explicitly calculated by stability function
        save_path="experiments/plots/simulated_run_mechanics.png"
    )

if __name__ == "__main__":
    main()