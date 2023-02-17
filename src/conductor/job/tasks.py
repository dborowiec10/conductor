

from conductor.orchestrator.default_orchestrator import DefaultOrchestrator
from conductor.executor.model_executor import ModelExecutor
from conductor.executor.tensor_program_executor import TensorProgramExecutor
from conductor.compiler.model_compiler import ModelCompiler
from conductor.compiler.tensor_program_compiler import TensorProgramCompiler

tasks = {
    "orchestrator:default": DefaultOrchestrator,
    "executor:model": ModelExecutor,
    "executor:tensor_program": TensorProgramExecutor,
    "compiler:model": ModelCompiler,
    "compiler:tensor_program": TensorProgramCompiler
}