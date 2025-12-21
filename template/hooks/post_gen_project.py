#!/usr/bin/env python
"""Post-generation hook for cookiecutter template."""

import os
import shutil
import subprocess
import sys

# Get cookiecutter variables
use_frontend = "{{ cookiecutter.use_frontend }}" == "True"
generate_env = "{{ cookiecutter.generate_env }}" == "True"
enable_i18n = "{{ cookiecutter.enable_i18n }}" == "True"

# Remove frontend folder if not using frontend
if not use_frontend:
    frontend_dir = os.path.join(os.getcwd(), "frontend")
    if os.path.exists(frontend_dir):
        shutil.rmtree(frontend_dir)
        print("Removed frontend/ directory (frontend not enabled)")

# Handle i18n disabled: move files from [locale]/ to app root
if use_frontend and not enable_i18n:
    app_dir = os.path.join(os.getcwd(), "frontend", "src", "app")
    locale_dir = os.path.join(app_dir, "[locale]")

    if os.path.exists(locale_dir):
        # Move all contents from [locale]/ to app/
        for item in os.listdir(locale_dir):
            src = os.path.join(locale_dir, item)
            dst = os.path.join(app_dir, item)
            # Skip the layout.tsx from [locale] - we'll use the root layout
            if item == "layout.tsx":
                continue
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)

        # Remove the now-empty [locale] directory
        shutil.rmtree(locale_dir)
        print("Moved routes from [locale]/ to app/ (i18n not enabled)")

        # Update root layout to include providers
        root_layout = os.path.join(app_dir, "layout.tsx")
        if os.path.exists(root_layout):
            with open(root_layout, "r") as f:
                content = f.read()
            # Add Providers import and wrap children
            content = content.replace(
                'import "./globals.css";',
                'import "./globals.css";\nimport { Providers } from "./providers";'
            )
            content = content.replace(
                "<body className={inter.className}>{children}</body>",
                "<body className={inter.className}>\n        <Providers>{children}</Providers>\n      </body>"
            )
            with open(root_layout, "w") as f:
                f.write(content)

    # Remove middleware.ts
    middleware_file = os.path.join(os.getcwd(), "frontend", "src", "middleware.ts")
    if os.path.exists(middleware_file):
        os.remove(middleware_file)

    # Remove i18n related files
    i18n_file = os.path.join(os.getcwd(), "frontend", "src", "i18n.ts")
    if os.path.exists(i18n_file):
        os.remove(i18n_file)

    messages_dir = os.path.join(os.getcwd(), "frontend", "messages")
    if os.path.exists(messages_dir):
        shutil.rmtree(messages_dir)

    # Remove language-switcher component
    lang_switcher = os.path.join(os.getcwd(), "frontend", "src", "components", "language-switcher.tsx")
    if os.path.exists(lang_switcher):
        os.remove(lang_switcher)

    print("Removed i18n files (i18n not enabled)")

# Remove .env files if generate_env is false
if not generate_env:
    backend_env = os.path.join(os.getcwd(), "backend", ".env")
    if os.path.exists(backend_env):
        os.remove(backend_env)
        print("Removed backend/.env (generate_env disabled)")

    frontend_env = os.path.join(os.getcwd(), "frontend", ".env.local")
    if os.path.exists(frontend_env):
        os.remove(frontend_env)
        print("Removed frontend/.env.local (generate_env disabled)")

# Generate uv.lock for backend (required for Docker builds)
backend_dir = os.path.join(os.getcwd(), "backend")
if os.path.exists(backend_dir):
    uv_cmd = shutil.which("uv")
    if uv_cmd:
        print("Generating uv.lock for backend...")
        result = subprocess.run(
            [uv_cmd, "lock"],
            cwd=backend_dir,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            print("uv.lock generated successfully.")
        else:
            print("Warning: Failed to generate uv.lock. Run 'uv lock' in backend/ directory.")
    else:
        print("Warning: uv not found. Run 'uv lock' in backend/ to generate lock file.")

# Run ruff to auto-fix import sorting and other linting issues
if os.path.exists(backend_dir):
    ruff_cmd = None

    # Try multiple methods to find/run ruff
    # 1. Check if ruff is in PATH
    ruff_path = shutil.which("ruff")
    if ruff_path:
        ruff_cmd = [ruff_path]
    # 2. Try uvx ruff (if uv is installed)
    elif shutil.which("uvx"):
        ruff_cmd = ["uvx", "ruff"]
    # 3. Try python -m ruff
    else:
        # Test if ruff is available as a module
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            ruff_cmd = [sys.executable, "-m", "ruff"]

    if ruff_cmd:
        print(f"Running ruff to format code (using: {' '.join(ruff_cmd)})...")
        # Run ruff check --fix to auto-fix issues
        subprocess.run(
            [*ruff_cmd, "check", "--fix", "--quiet", backend_dir],
            check=False,
        )
        # Run ruff format for consistent formatting
        subprocess.run(
            [*ruff_cmd, "format", "--quiet", backend_dir],
            check=False,
        )
        print("Code formatting complete.")
    else:
        print("Warning: ruff not found. Run 'ruff format .' in backend/ to format code.")

# Format frontend with prettier if it exists
frontend_dir = os.path.join(os.getcwd(), "frontend")
if use_frontend and os.path.exists(frontend_dir):
    # Try to find bun or npx for running prettier
    bun_cmd = shutil.which("bun")
    npx_cmd = shutil.which("npx")

    if bun_cmd:
        print("Installing frontend dependencies and formatting with Prettier...")
        # Install dependencies first (prettier is a devDependency)
        result = subprocess.run(
            [bun_cmd, "install"],
            cwd=frontend_dir,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            # Format with prettier
            subprocess.run(
                [bun_cmd, "run", "format"],
                cwd=frontend_dir,
                capture_output=True,
                check=False,
            )
            print("Frontend formatting complete.")
        else:
            print("Warning: Failed to install frontend dependencies.")
    elif npx_cmd:
        print("Formatting frontend with Prettier...")
        subprocess.run(
            [npx_cmd, "prettier", "--write", "."],
            cwd=frontend_dir,
            capture_output=True,
            check=False,
        )
        print("Frontend formatting complete.")
    else:
        print("Warning: bun/npx not found. Run 'bun run format' in frontend/ to format code.")

print("Project generated successfully!")
