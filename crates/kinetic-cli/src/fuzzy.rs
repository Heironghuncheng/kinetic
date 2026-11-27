//! Loose name matching, as people type model names: `qwen 4b` and `qwen4b`
//! both find `unsloth/Qwen3.5-4B-MTP-GGUF`.

/// How far `query` is from `text`, lower is closer: every word of the query
/// must appear in the text with its letters in order, each letter skipped in
/// between costs one. `None` when a word does not appear.
pub fn score(query: &str, text: &str) -> Option<usize> {
    let text = letters(text);
    query
        .split_whitespace()
        .map(|word| span(&letters(word), &text))
        .sum()
}

/// Lowercase letters and digits only, so `Qwen3.5-4B` reads as `qwen354b`.
fn letters(s: &str) -> Vec<char> {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// The fewest letters of `text` skipped to read `word` in it.
fn span(word: &[char], text: &[char]) -> Option<usize> {
    let first = *word.first()?;
    (0..text.len())
        .filter(|&start| text[start] == first)
        .filter_map(|start| {
            let mut at = start;
            for &c in &word[1..] {
                at += 1 + text.get(at + 1..)?.iter().position(|&t| t == c)?;
            }
            Some(at + 1 - start - word.len())
        })
        .min()
}

#[cfg(test)]
mod tests {
    use super::score;

    #[test]
    fn words_match_in_any_spacing_and_case() {
        let name = "unsloth/Qwen3.5-4B-MTP-GGUF";
        assert_eq!(score("qwen", name), Some(0));
        assert_eq!(score("qwen 4b", name), Some(0));
        assert_eq!(score("Qwen4B", name), Some(2));
        assert_eq!(score("", name), Some(0));
        assert_eq!(score("llama", name), None);
    }

    #[test]
    fn closer_spellings_score_lower() {
        let lfm = "LiquidAI/LFM3-1.7B-GGUF";
        assert!(score("lfm3", lfm) < score("lm3", lfm));
        assert_eq!(score("lfm 17", lfm), Some(0));
        assert_eq!(score("lfm4", lfm), None);
    }
}
