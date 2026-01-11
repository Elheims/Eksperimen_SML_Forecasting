from optuna_dashboard._cli import main as dashboard_main
import sys
import os

if __name__ == '__main__':
    if len(sys.argv) == 1:
        # Construct absolute path to the database to ensure it works from any working directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(script_dir, "optuna.db")
        db_url = f"sqlite:///{db_path}"
        
        print(f"Auto-configuring Optuna Dashboard...")
        print(f"Database: {db_url}")
        print(f"Port: 8080")
        
        # Inject arguments into sys.argv so cli.main() picks them up
        sys.argv.extend([db_url, "--port", "8080"])
    
    sys.exit(dashboard_main())
