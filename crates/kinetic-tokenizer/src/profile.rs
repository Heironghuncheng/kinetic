//! Vocabulary rules that depend on the GGUF architecture.

pub(crate) struct Profile {
    names: &'static [&'static str],
    pub trim_after: Option<fn(&str) -> bool>,
    pub extra_stops: &'static [&'static str],
}

const DEFAULT: Profile = Profile {
    names: &[],
    trim_after: None,
    extra_stops: &[],
};

const PROFILES: &[Profile] = &[Profile {
    names: &["phi3"],
    trim_after: Some(|text| !matches!(text, "<unk>" | "<s>" | "<|endoftext|>")),
    // The model can emit </s> even when the GGUF names a different EOS.
    extra_stops: &["</s>"],
}];

pub(crate) fn for_architecture(name: Option<&str>) -> &'static Profile {
    name.and_then(|name| PROFILES.iter().find(|p| p.names.contains(&name)))
        .unwrap_or(&DEFAULT)
}
