"""Run the frozen weighted, gated-routing, verification and summary pipeline."""
import run_weighted_loss_v5 as weighted
import run_routing_v5 as routing
import validate_improvements_v5 as validation
import summarize_improvements_v5 as summary

if __name__=='__main__':
    weighted.main()
    routing.main()
    validation.main()
    summary.main()
