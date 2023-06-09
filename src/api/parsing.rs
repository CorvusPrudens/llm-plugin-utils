#[derive(Default, Clone)]
pub enum JsonState {
    #[default]
    Idle,
    Active {
        data: String,
        num_brackets: usize,
        in_string: bool,
        escaped: bool,
    },
    MaybeIgnore {
        tick_count: usize,
    },
    Ignore {
        num_ticks: usize,
        tick_count: usize,
    },
}

pub fn parse_json_from_stream(
    input: &str,
    mut json_state: JsonState,
) -> (JsonState, Option<String>, String) {
    let mut completed_json = None;
    let mut filtered_delta = String::new();

    for ch in input.chars() {
        json_state = match json_state {
            JsonState::Idle => match ch {
                '{' => JsonState::Active {
                    data: "{".to_string(),
                    num_brackets: 1,
                    in_string: false,
                    escaped: false,
                },
                '`' => {
                    filtered_delta.push(ch);
                    JsonState::MaybeIgnore { tick_count: 1 }
                }
                _ => {
                    filtered_delta.push(ch);
                    JsonState::Idle
                }
            },
            JsonState::Active {
                mut data,
                num_brackets,
                in_string,
                escaped,
            } => {
                // Handle JSON string building
                match ch {
                    '{' if !in_string => {
                        data.push(ch);
                        JsonState::Active {
                            data,
                            num_brackets: num_brackets + 1,
                            in_string,
                            escaped,
                        }
                    }
                    '}' if !in_string => {
                        let num_brackets = num_brackets - 1;
                        data.push(ch);
                        if num_brackets == 0 {
                            // We've finished reading the JSON object
                            completed_json = Some(data);
                            JsonState::Idle
                        } else {
                            JsonState::Active {
                                data,
                                num_brackets,
                                in_string,
                                escaped,
                            }
                        }
                    }
                    '"' if in_string && !escaped => {
                        data.push(ch);
                        JsonState::Active {
                            data,
                            num_brackets,
                            in_string: false,
                            escaped,
                        }
                    }
                    '"' if !in_string && !escaped => {
                        data.push(ch);
                        JsonState::Active {
                            data,
                            num_brackets,
                            in_string: true,
                            escaped,
                        }
                    }
                    '\\' if !escaped => {
                        // If we encounter a backslash and the previous character wasn't a backslash
                        // Set escaped flag
                        JsonState::Active {
                            data,
                            num_brackets,
                            in_string,
                            escaped: true,
                        }
                        // Do not push backslash character to data yet
                        // It will be pushed in next iteration if necessary (when escaped character is not a quote)
                    }
                    _ => {
                        // Reset escaped flag (if it was set)
                        // Push other characters as they are part of JSON
                        data.push(ch);
                        JsonState::Active {
                            data,
                            num_brackets,
                            in_string,
                            escaped: false,
                        }
                    }
                }
            }
            JsonState::MaybeIgnore { tick_count } => {
                filtered_delta.push(ch);
                match ch {
                    '`' => JsonState::MaybeIgnore {
                        tick_count: tick_count + 1,
                    },
                    _ => JsonState::Ignore {
                        num_ticks: tick_count,
                        tick_count: 0,
                    },
                }
            }
            JsonState::Ignore {
                num_ticks,
                tick_count,
            } => {
                filtered_delta.push(ch);
                match ch {
                    '`' => {
                        let tick_count = tick_count + 1;
                        if tick_count == num_ticks {
                            // We found the closing sequence
                            JsonState::Idle
                        } else {
                            JsonState::Ignore {
                                num_ticks,
                                tick_count,
                            }
                        }
                    }
                    _ => {
                        // Any non-backtick character resets the count
                        JsonState::Ignore {
                            num_ticks,
                            tick_count: 0,
                        }
                    }
                }
            }
        };
    }

    (json_state, completed_json, filtered_delta)
}
