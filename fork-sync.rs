#!/usr/bin/env -S cargo +nightly -Zscript
---
[package]
name = "fork-sync"
version = "0.1.0"
edition = "2021"

[dependencies]
toml = "0.9"
---

//! flatland fork maintenance: sparse-checkout + workspace pruning, idempotent.
//!
//! Run after every upstream bump (rebase onto new base):
//!
//!     cargo +nightly -Zscript fork-sync.rs --base 0.85.0
//!
//! What it does:
//!   1. Reads workspace members + path-dep graph FROM THE BASE GIT REF
//!      (`git cat-file`), never from the working tree — safe to run on an
//!      inconsistent/sparse worktree mid-bump.
//!   2. Computes the transitive closure of the crates we actually modify
//!      (WANT list below) over normal+dev path deps.
//!   3. Writes .git/info/sparse-checkout: root files + closure dirs only.
//!   4. Prunes root Cargo.toml `members` to the closure (the ONE tracked
//!      change; re-run after rebase to regenerate deterministically).
//!   5. Restores Cargo.lock verbatim from the base ref — no resolution
//!      drift, ever.
//!   6. Applies sparsity and verifies: cargo metadata must succeed.
//!
//! To add/remove forked crates, edit WANT and re-run.

use std::collections::{BTreeMap, BTreeSet};
use std::process::Command;

/// Crates we actually fork/modify. Everything else stays upstream-at-rest.
const WANT: &[&str] = &["vortex-array", "vortex-compute", "encodings/fastlanes", "encodings/zigzag"];

fn git(args: &[&str]) -> String {
    let out = Command::new("git").args(args).output().unwrap_or_else(|e| {
        panic!("git {args:?} failed to spawn: {e}");
    });
    if !out.status.success() {
        panic!("git {args:?} failed:\n{}", String::from_utf8_lossy(&out.stderr));
    }
    String::from_utf8(out.stdout).expect("git output utf8")
}

/// Parse member manifests from the base ref; return member -> set of path-dep
/// members. Deps may be declared inline (`path = "../x"`) or inherited
/// (`workspace = true`, paths living in root `[workspace.dependencies]`).
fn dep_graph(
    base: &str,
    members: &[String],
    ws_deps: &BTreeMap<String, String>,
) -> BTreeMap<String, BTreeSet<String>> {
    let member_set: BTreeSet<&String> = members.iter().collect();
    let mut graph = BTreeMap::new();
    for m in members {
        let blob = Command::new("git")
            .args(["cat-file", "blob", &format!("{base}:{m}/Cargo.toml")])
            .output()
            .unwrap();
        if !blob.status.success() {
            continue; // member without manifest at base — skip
        }
        let v: toml::Value = toml::from_str(&String::from_utf8_lossy(&blob.stdout))
            .unwrap_or_else(|e| panic!("{m}/Cargo.toml at {base}: {e}"));
        let mut deps = BTreeSet::new();
        for table_key in ["dependencies", "dev-dependencies", "build-dependencies"] {
            if let Some(t) = v.get(table_key).and_then(|t| t.as_table()) {
                for (name, spec) in t {
                    // inline path dep
                    if let Some(p) = spec.get("path").and_then(|p| p.as_str()) {
                        let dep_dir = normalize(&format!("{m}/{p}"));
                        if member_set.contains(&dep_dir) {
                            deps.insert(dep_dir);
                        }
                    } else if spec.get("workspace") == Some(&toml::Value::Boolean(true)) {
                        // workspace-inherited: resolve via root [workspace.dependencies]
                        if let Some(ws_path) = ws_deps.get(name) {
                            let dep_dir = normalize(ws_path);
                            if member_set.contains(&dep_dir) {
                                deps.insert(dep_dir);
                            }
                        }
                    }
                }
            }
        }
        graph.insert(m.clone(), deps);
    }
    graph
}

/// Normalize "a/../b/./c" -> "b/c"
fn normalize(p: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for seg in p.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            s => out.push(s),
        }
    }
    out.join("/")
}

fn main() {
    let mut base = String::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--base" => base = args.next().expect("--base needs value"),
            other => panic!("unknown arg {other}; usage: fork-sync.rs --base <ref>"),
        }
    }
    if base.is_empty() {
        panic!("usage: fork-sync.rs --base <ref> [--check]");
    }

    // 1. Members + workspace.dependencies path map from base root manifest.
    let root_toml: toml::Value =
        toml::from_str(&git(&["cat-file", "blob", &format!("{base}:Cargo.toml")])).unwrap();
    let members: Vec<String> = root_toml["workspace"]["members"]
        .as_array()
        .expect("workspace.members")
        .iter()
        .map(|x| x.as_str().expect("member str").to_string())
        .collect();
    let ws_deps: BTreeMap<String, String> = root_toml["workspace"]
        .get("dependencies")
        .and_then(|d| d.as_table())
        .map(|t| {
            t.iter()
                .filter_map(|(name, spec)| {
                    spec.get("path")
                        .and_then(|p| p.as_str())
                        .map(|p| (name.clone(), p.to_string()))
                })
                .collect()
        })
        .unwrap_or_default();

    // 2. Closure over path deps.
    let graph = dep_graph(&base, &members, &ws_deps);
    let mut keep: BTreeSet<String> = WANT.iter().map(|s| s.to_string()).collect();
    let mut stack: Vec<String> = keep.iter().cloned().collect();
    while let Some(d) = stack.pop() {
        if let Some(deps) = graph.get(&d) {
            for dd in deps {
                if keep.insert(dd.clone()) {
                    stack.push(dd.clone());
                }
            }
        } else {
            panic!("want/closure crate `{d}` has no manifest in {base} members — typo?");
        }
    }
    println!(
        "closure of {:?} @ {} = {} crates:\n  {}",
        WANT,
        base,
        keep.len(),
        keep.iter().cloned().collect::<Vec<_>>().join("\n  ")
    );

    // 3. Sparse patterns: all root files, only closure dirs beneath.
    let mut patterns = String::from("/*\n!/*/\n");
    for k in &keep {
        patterns.push_str(&format!("/{k}/\n"));
    }
    std::fs::write(".git/info/sparse-checkout", &patterns).unwrap();

    // 4. Prune worktree Cargo.toml members to closure (deterministic rewrite).
    let cur = std::fs::read_to_string("Cargo.toml").unwrap();
    let start = cur.find("members = [").expect("members array in root Cargo.toml");
    let end_rel = cur[start..].find("]").expect("members array close");
    let pruned = format!(
        "# fork-sync: pruned to flatland closure (rerun `cargo -Zscript fork-sync.rs --base <ref>` after upstream bumps)\nmembers = [\n{}\n]",
        keep.iter()
            .map(|k| format!("    \"{k}\","))
            .collect::<Vec<_>>()
            .join("\n")
    );
    let mut new_root = String::with_capacity(cur.len());
    new_root.push_str(&cur[..start]);
    new_root.push_str(&pruned);
    new_root.push_str(&cur[start + end_rel + 1..]);
    std::fs::write("Cargo.toml", new_root).unwrap();

    // 5. Lock verbatim from base — zero resolution drift.
    std::fs::write("Cargo.lock", git(&["cat-file", "blob", &format!("{base}:Cargo.lock")]))
        .unwrap();

    // 6. Apply sparsity + verify.
    git_silent(&["config", "core.sparseCheckout", "true"]);
    apply_sparse();
    let meta = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()
        .unwrap();
    assert!(
        meta.status.success(),
        "post-sync cargo metadata failed:\n{}",
        String::from_utf8_lossy(&meta.stderr)
    );
    println!(
        "\nok: {} members active, sparse applied.\nCommit the Cargo.toml members prune:\n  jj describe / git commit -am 'flatland: sync sparse workspace to {base}'",
        keep.len()
    );
}

fn git_silent(args: &[&str]) {
    Command::new("git").args(args).output().expect("git spawn");
}

fn apply_sparse() {
    let patterns = std::fs::read_to_string(".git/info/sparse-checkout").unwrap();
    use std::io::Write;
    let mut child = Command::new("git")
        .args(["sparse-checkout", "set", "--no-cone", "--stdin"])
        .stdin(std::process::Stdio::piped())
        .spawn()
        .expect("sparse-checkout spawn");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(patterns.as_bytes())
        .unwrap();
    let st = child.wait().expect("sparse-checkout wait");
    assert!(st.success(), "git sparse-checkout set failed");
}
