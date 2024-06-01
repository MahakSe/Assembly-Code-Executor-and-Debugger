# Assembly Code Executor and Debugger

## Overview
The Assembly Code Executor and Debugger is a Python-based tool designed for interpreting and debugging assembly code. It executes assembly instructions across data, code, and stack segments, providing detailed feedback on the state of registers, flags, stack, and memory after each line of code execution. This tool enhances the debugging process by offering a clear view of memory bytes and execution states.

## Features
- **Assembly Code Interpretation**: Execute assembly instructions within data, code, and stack segments.
- **Debugging Capabilities**: Step-by-step execution with detailed feedback on registers, flags, stack, and memory.
- **Memory Inspection**: View memory contents in bytes after each instruction execution.
- **Real-Time Updates**: Display real-time status of registers and flags during code execution.
- **User-Friendly Interface**: Command-line interface for ease of use.

## How It Works

### Architecture Overview
The Assembly Code Executor and Debugger is structured into several key components: the interpreter, the debugger, and utility functions. Each component interacts with others to provide a comprehensive debugging tool.

### Key Components

1. **Interpreter**:
    - **Execution**: Parses and executes assembly instructions.
    - **Segment Handling**: Manages data, code, and stack segments during execution.

2. **Debugger**:
    - **Step-by-Step Execution**: Allows users to execute code one instruction at a time.
    - **State Display**: Shows the current state of registers, flags, stack, and memory after each instruction.

3. **Memory Inspection**:
    - Provides a detailed view of memory contents in bytes.
    - Allows users to monitor changes in memory during code execution.

### Example Workflow
1. **Loading Assembly Code**:
    - The user loads an assembly code file into the system.

2. **Executing Code**:
    - The user starts execution, and the system processes each instruction, updating and displaying the state.

3. **Inspecting State**:
    - After each instruction, the user can inspect the current state of registers, flags, stack, and memory.

By following this structured approach, the Assembly Code Executor and Debugger provides a robust and user-friendly platform for debugging assembly code.
