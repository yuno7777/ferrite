//! LM Studio interop.
//!
//! LM Studio already downloaded your models; there is no reason for ferrite to
//! download them again. It lays them out as
//! `<models>/<publisher>/<repo>/<file>.gguf`, which becomes the model id
//! `publisher/repo/file`.

use std::env;
use std::fs;
use std::io::{Error, ErrorKind, Result};
use std::path::{Path, PathBuf};

/// How deep to walk under a models directory. Real layout is 2 levels; a few
/// more costs nothing and survives people who nest by quant level.
const MAX_DEPTH: u32 = 5;

#[derive(Clone, Debug)]
pub struct Model {
    /// `publisher/repo/file`, no extension.
    pub id: String,
    pub path: PathBuf,
    pub size: u64,
}

fn home() -> Option<PathBuf> {
    env::var_os("USERPROFILE")
        .or_else(|| env::var_os("HOME"))
        .map(PathBuf::from)
}

/// Every directory we look in, most specific first. `FERRITE_MODELS_DIR` wins
/// so a custom store works without LM Studio installed at all.
pub fn model_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for var in ["FERRITE_MODELS_DIR", "LMSTUDIO_MODELS_DIR"] {
        if let Some(dir) = env::var_os(var) {
            dirs.push(PathBuf::from(dir));
        }
    }
    if let Some(home) = home() {
        // 0.3 and later.
        dirs.push(home.join(".lmstudio").join("models"));
        // 0.2 and earlier.
        dirs.push(home.join(".cache").join("lm-studio").join("models"));
    }
    dirs.retain(|d| d.is_dir());
    dirs.dedup();
    dirs
}

/// Every GGUF under every known models directory, sorted by id.
pub fn list() -> Vec<Model> {
    let mut out = Vec::new();
    for root in model_dirs() {
        walk(&root, &root, 0, &mut out);
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out.dedup_by(|a, b| a.path == b.path);
    out
}

fn walk(root: &Path, dir: &Path, depth: u32, out: &mut Vec<Model>) {
    if depth > MAX_DEPTH {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(ty) = entry.file_type() else { continue };
        // Not following links keeps a symlinked models dir from looping.
        if ty.is_symlink() {
            continue;
        }
        let path = entry.path();
        if ty.is_dir() {
            walk(root, &path, depth + 1, out);
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            out.push(Model {
                id: id_for(root, &path),
                path,
                size,
            });
        }
    }
}

fn id_for(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let rel = rel.with_extension("");
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// A path, or a substring of a model id. Ambiguity is an error rather than a
/// coin flip — loading the wrong 4 GB file wastes a minute of your life.
pub fn resolve(query: &str) -> Result<PathBuf> {
    let direct = Path::new(query);
    if direct.is_file() {
        return Ok(direct.to_path_buf());
    }

    let needle = query.to_lowercase();
    let models = list();
    let hits: Vec<&Model> = models
        .iter()
        .filter(|m| m.id.to_lowercase().contains(&needle))
        .collect();

    match hits.as_slice() {
        [] if models.is_empty() => Err(Error::new(
            ErrorKind::NotFound,
            format!(
                "no models found. Looked in: {}. Point at a .gguf path, or set FERRITE_MODELS_DIR.",
                dirs_hint()
            ),
        )),
        [] => Err(Error::new(
            ErrorKind::NotFound,
            format!("no model id matches {query:?}. Run `ferrite list`."),
        )),
        [one] => Ok(one.path.clone()),
        many => {
            let names: Vec<&str> = many.iter().map(|m| m.id.as_str()).collect();
            Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "{query:?} matches {} models: {}",
                    many.len(),
                    names.join(", ")
                ),
            ))
        }
    }
}

fn dirs_hint() -> String {
    let dirs = model_dirs();
    if dirs.is_empty() {
        return "(none exist)".into();
    }
    dirs.iter()
        .map(|d| d.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_is_publisher_repo_file() {
        let root = Path::new("/models");
        let path = Path::new("/models/lmstudio-community/Llama-3.2-1B/model-Q4_K_M.gguf");
        assert_eq!(
            id_for(root, path),
            "lmstudio-community/Llama-3.2-1B/model-Q4_K_M"
        );
    }
}
