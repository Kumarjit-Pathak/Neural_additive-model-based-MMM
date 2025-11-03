#!/usr/bin/env python3
"""
Main orchestrator for NAM agent-based development
Runs all agents according to workflow
"""
import asyncio
import yaml
import json
from pathlib import Path
import logging
from datetime import datetime


class AgentOrchestrator:
    """Orchestrator for agent-based development"""

    def __init__(self, workflow_path='.agents/orchestrator/workflow.yaml'):
        self.workflow_path = Path(workflow_path)
        self.workflow = self.load_workflow()
        self.agent_states = {}
        self.logger = self.setup_logging()

    def load_workflow(self):
        """Load workflow configuration"""
        with open(self.workflow_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Setup logging"""
        log_file = '.agents/orchestrator/orchestrator.log'
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('Orchestrator')

    async def execute_workflow(self):
        """Execute complete workflow"""
        self.logger.info("="*70)
        self.logger.info("Starting NAM Agent-Based Development Workflow")
        self.logger.info("="*70)

        for phase in self.workflow['workflow']['phases']:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Phase: {phase['name']}")
            self.logger.info(f"Description: {phase['description']}")
            self.logger.info(f"Parallel: {phase['parallel']}")
            self.logger.info(f"{'='*70}")

            if phase['parallel']:
                await self.execute_phase_parallel(phase)
            else:
                await self.execute_phase_sequential(phase)

            # Check phase completion
            if not self.check_phase_completion(phase):
                self.logger.error(f"Phase {phase['name']} failed!")
                return False

            self.logger.info(f"Phase {phase['name']} completed successfully")

        self.logger.info("\n" + "="*70)
        self.logger.info("All phases completed successfully!")
        self.logger.info("="*70)
        return True

    async def execute_phase_parallel(self, phase):
        """Execute phase with parallel agents"""
        tasks = []

        for agent_config in phase['agents']:
            agent_name = list(agent_config.keys())[0]
            task_name = agent_config[agent_name]

            task = self.run_agent_task(agent_name, task_name)
            tasks.append(task)

        # Run all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log results
        for agent_config, result in zip(phase['agents'], results):
            agent_name = list(agent_config.keys())[0]
            if isinstance(result, Exception):
                self.logger.error(f"Agent {agent_name} failed: {result}")
            else:
                self.logger.info(f"Agent {agent_name} completed successfully")

    async def execute_phase_sequential(self, phase):
        """Execute phase with sequential agents"""
        for agent_config in phase['agents']:
            agent_name = list(agent_config.keys())[0]
            task_name = agent_config[agent_name]

            result = await self.run_agent_task(agent_name, task_name)

            if isinstance(result, Exception):
                self.logger.error(f"Agent {agent_name} failed: {result}")
                return False

    async def run_agent_task(self, agent_name, task_name):
        """Run a specific agent task"""
        self.logger.info(f"Running {agent_name}.{task_name}")

        # For now, this is a placeholder that marks tasks as complete
        # In a real implementation, this would call agent-specific Python scripts

        # Simulate task execution
        await asyncio.sleep(0.1)

        # Update state
        self.update_agent_state(agent_name, task_name, 'completed')

        return True

    def check_phase_completion(self, phase):
        """Check if all agents in phase completed"""
        for agent_config in phase['agents']:
            agent_name = list(agent_config.keys())[0]
            task_name = agent_config[agent_name]

            state = self.agent_states.get(agent_name, {}).get(task_name, 'pending')
            if state != 'completed':
                return False

        return True

    def update_agent_state(self, agent_name, task_name, state):
        """Update agent state"""
        if agent_name not in self.agent_states:
            self.agent_states[agent_name] = {}

        self.agent_states[agent_name][task_name] = state

        # Save state to file
        state_path = f'.agents/orchestrator/state.json'
        Path(state_path).parent.mkdir(parents=True, exist_ok=True)

        with open(state_path, 'w') as f:
            json.dump(self.agent_states, f, indent=2)


async def main():
    """Main entry point"""
    orchestrator = AgentOrchestrator()
    success = await orchestrator.execute_workflow()

    if success:
        print("\n🎉 Agent-based development completed successfully!")
    else:
        print("\n❌ Development workflow failed")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
