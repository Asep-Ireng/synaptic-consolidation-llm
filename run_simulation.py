import sys
import warnings
from src.model import LivingLLM
from src.memory import load_ltm_to_stm

# Suppress messy warnings from libraries
warnings.filterwarnings("ignore")

# ANSI Colors for cleaner terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

def main():
    print(f"{Colors.HEADER}--- SYNAPTIC CONSOLIDATION EXPERIMENT (Bio-LLM) ---{Colors.ENDC}")
    print("Loading Cortex... (This may take a moment)")

    try:
        # Initialize
        brain = LivingLLM(MODEL_NAME)
        load_ltm_to_stm(brain)
        print(f"{Colors.GREEN}✔ System Online.{Colors.ENDC}")
        print(f"\n{Colors.YELLOW}Commands:{Colors.ENDC}")
        print(f"  {Colors.BLUE}!learn [text]{Colors.ENDC} -> Force-feed a new fact (Wake State)")
        print(f"  {Colors.BLUE}!sleep{Colors.ENDC}        -> Trigger dream cycle & pruning (Sleep State)")
        print(f"  {Colors.BLUE}!save{Colors.ENDC}         -> Save current brain state")
        print(f"  {Colors.BLUE}!exit{Colors.ENDC}         -> Quit")
        
    except Exception as e:
        print(f"{Colors.FAIL}CRITICAL FAILURE DURING LOAD: {e}{Colors.ENDC}")
        return

    while True:
        try:
            # User Input
            user_input = input(f"\n{Colors.HEADER}User:{Colors.ENDC} ").strip()
            
            if not user_input: continue
            
            # --- COMMAND HANDLER ---
            if user_input.lower() in ["!exit", "quit", "exit"]:
                print("Shutting down...")
                break
                
            elif user_input.startswith("!learn"):
                # Wake State
                fact = user_input.replace("!learn", "", 1).strip()
                if not fact:
                    print(f"{Colors.YELLOW}⚠ Usage: !learn [sentence]{Colors.ENDC}")
                    continue
                brain.learn(fact, steps=5)
                
            elif user_input.startswith("!sleep"):
                # Sleep State
                brain.sleep()

            elif user_input.startswith("!save"):
                # Optional: Helper to save manually
                from src.utils import save_checkpoint
                save_checkpoint(brain, "checkpoints/manual_save.pt", brain.optimizer)

            else:
                # Inference State
                # We wrap inference in a try-block too, just in case context length overflows
                response = brain.talk(user_input)
                print(f"{Colors.GREEN}BioLLM:{Colors.ENDC} {response}")

        except KeyboardInterrupt:
            # Catches Ctrl+C so you can abort a long generation without killing the app
            print(f"\n{Colors.YELLOW}⚠ Interrupted by user.{Colors.ENDC}")
            continue
            
        except RuntimeError as e:
            # Catches CUDA OOM errors specifically
            if "out of memory" in str(e).lower():
                print(f"{Colors.FAIL}❌ GPU OUT OF MEMORY! Attempting to clear cache...{Colors.ENDC}")
                import torch
                torch.cuda.empty_cache()
            else:
                print(f"{Colors.FAIL}❌ Runtime Error: {e}{Colors.ENDC}")
                
        except Exception as e:
            print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")

if __name__ == "__main__":
    main()