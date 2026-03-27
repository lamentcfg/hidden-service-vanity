/// Configuration for ntfy.sh notifications
pub struct NtfyConfig {
    pub host: String,
    pub topic: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub on_match: bool,
}

impl NtfyConfig {
    pub fn is_enabled(&self) -> bool {
        !self.topic.is_empty()
    }

    fn build_client(&self) -> reqwest::blocking::RequestBuilder {
        let client = reqwest::blocking::Client::new();
        let url = format!("{}/{}", self.host.trim_end_matches('/'), self.topic);
        let mut builder = client.post(&url);

        if let (Some(user), Some(pass)) = (&self.username, &self.password) {
            builder = builder.basic_auth(user, Some(pass));
        }

        builder
    }
}

pub fn send_ntfy_notification(config: &NtfyConfig, title: &str, message: &str) {
    let result = config
        .build_client()
        .header("Title", title)
        .header("Priority", "default")
        .body(message.to_string())
        .send();

    match result {
        Ok(resp) => {
            if !resp.status().is_success() {
                eprintln!("  [ntfy] Notification failed: HTTP {}", resp.status());
            }
        }
        Err(e) => {
            eprintln!("  [ntfy] Notification failed: {}", e);
        }
    }
}
