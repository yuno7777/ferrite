use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::synth::Builder;
use ferrite::template::{ChatMessage, ChatTemplate};

#[test]
fn test_chatml_template_generation() {
    let template = ChatTemplate::chatml();
    let messages = vec![
        ChatMessage::system("You are a helpful coding assistant."),
        ChatMessage::user("Write a function in Rust."),
        ChatMessage::assistant("Here is the function: fn add(a: i32, b: i32) -> i32 { a + b }"),
        ChatMessage::user("Now add documentation."),
    ];

    let prompt_with_gen = template.render(&messages, true, None, None);
    assert!(prompt_with_gen
        .starts_with("<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n"));
    assert!(prompt_with_gen.contains("<|im_start|>user\nWrite a function in Rust.<|im_end|>\n"));
    assert!(prompt_with_gen.contains("<|im_start|>assistant\nHere is the function: fn add(a: i32, b: i32) -> i32 { a + b }<|im_end|>\n"));
    assert!(prompt_with_gen.contains("<|im_start|>user\nNow add documentation.<|im_end|>\n"));
    assert!(prompt_with_gen.ends_with("<|im_start|>assistant\n"));

    let prompt_without_gen = template.render(&messages, false, None, None);
    assert!(!prompt_without_gen.ends_with("<|im_start|>assistant\n"));
    assert!(prompt_without_gen.ends_with("<|im_end|>\n"));
}

#[test]
fn test_llama3_template_generation() {
    let template = ChatTemplate::llama3();
    let messages = vec![
        ChatMessage::system("You are an AI."),
        ChatMessage::user("Hello!"),
    ];

    let rendered = template.render(&messages, true, Some("<|begin_of_text|>"), None);
    assert_eq!(
        rendered,
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an AI.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    );
}

#[test]
fn test_llama2_inst_template_generation() {
    let template = ChatTemplate::llama2();
    let messages = vec![
        ChatMessage::system("System instructions."),
        ChatMessage::user("User question."),
    ];

    let rendered = template.render(&messages, false, None, None);
    assert!(rendered.contains("<<SYS>>\nSystem instructions.\n<</SYS>>\n\n"));
    assert!(rendered.contains("[INST] User question. [/INST]"));
}

#[test]
fn test_gemma_template_generation() {
    let template = ChatTemplate::gemma();
    let messages = vec![
        ChatMessage::user("Hello Gemma"),
        ChatMessage::assistant("Hello user"),
    ];

    let rendered = template.render(&messages, true, None, None);
    assert_eq!(
        rendered,
        "<start_of_turn>user\nHello Gemma<end_of_turn>\n<start_of_turn>assistant\nHello user<end_of_turn>\n<start_of_turn>assistant\n"
    );
}

#[test]
fn test_custom_template_with_conditionals_and_loop_helpers() {
    let custom = r#"
{%- for msg in messages %}
[{{ loop.index }}/{{ loop.index0 }}] {{ msg.role | trim }}: {{ msg.content | trim }}
{%- if loop.first %} (FIRST){% endif %}
{%- if loop.last %} (LAST){% endif %}
{% endfor %}
{%- if not add_generation_prompt %}
[DONE]
{%- else %}
[NEXT: assistant]
{%- endif %}"#;

    let template = ChatTemplate::parse(custom);
    let messages = vec![
        ChatMessage::user(" First question "),
        ChatMessage::assistant(" First answer "),
    ];

    let rendered_with_gen = template.render(&messages, true, None, None);
    assert!(rendered_with_gen.contains("[1/0] user: First question (FIRST)"));
    assert!(rendered_with_gen.contains("[2/1] assistant: First answer (LAST)"));
    assert!(rendered_with_gen.contains("[NEXT: assistant]"));

    let rendered_without_gen = template.render(&messages, false, None, None);
    assert!(rendered_without_gen.contains("[DONE]"));
}

#[test]
fn test_template_loaded_from_gguf_metadata() {
    let mut temp = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    temp.push(format!("ferrite_chat_meta_{nanos}.gguf"));

    let custom_jinja =
        "{% for m in messages %}<turn>{{ m.role }}: {{ m.content }}</turn>{% endfor %}";

    let builder = Builder::new()
        .meta("general.architecture", Value::String("custom".into()))
        .meta(
            "tokenizer.chat_template",
            Value::String(custom_jinja.into()),
        )
        .tensor("dummy", &[32], GgmlType::F32, vec![0u8; 128]);
    builder.write(&temp).unwrap();

    let gguf = Gguf::open(&temp).unwrap();
    let template_str = gguf.meta_str("tokenizer.chat_template").unwrap();
    let template = ChatTemplate::parse(template_str);

    let messages = vec![ChatMessage::user("Hi"), ChatMessage::assistant("Hello")];
    let rendered = template.render(&messages, false, None, None);
    assert_eq!(
        rendered,
        "<turn>user: Hi</turn><turn>assistant: Hello</turn>"
    );

    let _ = std::fs::remove_file(temp);
}
