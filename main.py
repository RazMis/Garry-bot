import random
import sys
import threading
from difflib import get_close_matches
import time
from datetime import datetime, timedelta
from collections import deque
import os
import traceback
import requests

import speech_recognition
import pyttsx3 as tts
import json
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io
import tempfile


class ConversationMemory:
    def __init__(self, max_turns=5, memory_duration=300):
        self.memory = deque(maxlen=max_turns)
        self.memory_duration = memory_duration

    def add_exchange(self, user_input, assistant_response):
        timestamp = datetime.now()
        self.memory.append({
            'user_input': user_input,
            'assistant_response': assistant_response,
            'timestamp': timestamp
        })

    def get_recent_context(self):
        # Remove old memories more efficiently
        current_time = datetime.now()
        # Use list comprehension instead of creating a new deque
        old_memories = [m for m in self.memory
                        if (current_time - m['timestamp']).total_seconds() <= self.memory_duration]

        # Only create a new deque if needed
        if len(old_memories) != len(self.memory):
            self.memory = deque(old_memories, maxlen=self.memory.maxlen)

        # Return early if memory is empty
        if not self.memory:
            return ""

        # Format context for the LLM
        context_parts = []
        for exchange in self.memory:
            context_parts.append(f"Human: {exchange['user_input']}")
            context_parts.append(f"Tsun: {exchange['assistant_response']}")

        return "\n".join(context_parts)


class Assistant:
    # Class constants
    EXIT_COMMANDS = {
        'stop', 'sleep', 'goodbye', 'bye', 'exit', 'quit',
        'shut down', 'cancel', 'end', 'terminate', 'good night'
    }

    WAKE_WORDS = {"hey there", "hey mizu", "hello there", "hi guest", "hello guest", "hello"}

    # Default configurations - moved to class constants
    DEFAULT_CONFIG = {
        "voice_index": 0,  # 0 for Microsoft David (male), 1 for Microsoft Zira (female)
        "speech_rate": 170,  # Base speech rate for pyttsx3 (default 200)
        "volume": 1.0,  # Volume from 0.0 to 1.0
        "semitones": -5,  # Pitch adjustment (-12 to +12 recommended range)
        "speed": 1.0,  # Speed adjustment (1.0 = normal, 0.5 = half, 2.0 = double)
        "locale": "en-us",  # Language locale
        "recognition_energy": 1000,  # Recognition energy threshold
        "pause_threshold": 2.0,  # Speech recognition pause threshold
        "phrase_threshold": 0.5,  # Speech recognition phrase threshold
        "non_speaking_duration": 1.0,  # Non-speaking duration
        "deepseek_api_endpoint": "https://api.deepseek.com/v1/chat/completions",
        "model_name": "deepseek-chat",  # Default model name
        "temperature": 0.7,  # Temperature for generation
        "max_tokens": 200  # Maximum tokens for response
    }

    # Default personality prompt
    DEFAULT_PERSONALITY = """You are Tsun, a Mizutsune from the Monster Hunter series..."""

    def __init__(self):
        # Initialize conversation memory
        self.memory = ConversationMemory(max_turns=3, memory_duration=180)

        # Initialize speech recognition
        self.recogniser = speech_recognition.Recognizer()

        # Import audio libraries to class attributes
        self.gTTS = gTTS
        self.AudioSegment = AudioSegment
        self.play = play
        self.io = io
        self.tempfile = tempfile

        # Set default state variables
        self.is_speaking = False
        self.running = True

        # Set up configuration
        self.load_personality("personality.txt")
        self.load_audio_config("audio_config.json")

        # Common responses cache for quick replies
        self.setup_response_cache()

        # DeepSeek API configuration
        self.deepseek_api_key = "sk-6576877a59284e89a461254b3af4851a"  # Replace with your actual API key
        self.deepseek_api_endpoint = "https://api.deepseek.com/v1/chat/completions"
        self.deepseek_model_name = "deepseek-chat"  # Replace with the correct model name

        # Start the assistant
        try:
            self.run_assistant()
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.cleanup()

    def setup_response_cache(self):
        """Set up cache for common responses."""
        self.response_cache = {
            "how are you": "I'm good.",
            "hello": "Hey there! What are you doing?",
            "hi": "Hi!",
            "what's your name": "I'm Tsun, the mighty and beautiful Mizutsune.",
            "goodbye": "Bye bye!",
            "bye": "Catch you later! Stay clean!",
        }

    def load_personality(self, file_path):
        """Load the personality prompt from a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                self.personality_prompt = file.read().strip()
            print(f"Personality loaded from {file_path}")
        except FileNotFoundError:
            print(f"Personality file {file_path} not found, using default personality")
            # Use default personality as fallback
            self.personality_prompt = self.DEFAULT_PERSONALITY

    def load_audio_config(self, file_path):
        """Load audio configuration settings from a JSON file."""
        try:
            with open(file_path, "r") as file:
                config = json.load(file)

            # Create a copy of default config
            loaded_config = self.DEFAULT_CONFIG.copy()

            # Update with values from file
            for key, value in config.items():
                loaded_config[key] = value

            print(f"Audio configuration loaded from {file_path}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading audio config: {e}. Using default settings.")
            loaded_config = self.DEFAULT_CONFIG.copy()

        # Apply all settings to self
        for key, value in loaded_config.items():
            setattr(self, key, value)

        # Update recognizer settings
        self.recogniser.energy_threshold = self.recognition_energy
        self.recogniser.pause_threshold = self.pause_threshold
        self.recogniser.phrase_threshold = self.phrase_threshold
        self.recogniser.non_speaking_duration = self.non_speaking_duration

        # Initialize Ollama API endpoints after loading config
        # Print the current audio settings
        print(f"Voice: {'Male' if self.voice_index == 0 else 'Female'}, "
              f"Pitch: {self.semitones} semitones, "
              f"Speed: {self.speed}x, "
              f"Volume: {self.volume}")

    def reload_config(self):
        """Reload configuration files."""
        self.load_personality("personality.txt")
        self.load_audio_config("audio_config.json")

    def call_deepseek_api(self, prompt, max_tokens=150):
        """Call the DeepSeek API with the given prompt."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.deepseek_api_key}"
        }

        data = {
            "model": self.deepseek_model_name,
            "messages": [
                {"role": "system", "content": self.personality_prompt},  # Use the loaded personality prompt
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            response = requests.post(self.deepseek_api_endpoint, headers=headers, json=data)
            response.raise_for_status()  # Raise an exception for error status codes

            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            print(f"Error calling DeepSeek API: {e}")
            return "Oops, something went wrong. Let me try again."

    def get_llm_response(self, text):
        """Get a response using the cache or the LLM."""
        try:
            # First check the response cache for instant replies
            text_lower = text.lower()
            if text_lower in self.response_cache:
                return self.response_cache[text_lower]

            # Use the LLM for all other responses
            return self.generate_llm_response(text)  # Only pass `text`
        except Exception as e:
            print(f"Error generating response: {e}")
            # Fallback response to prevent crashes
            return "Oops, I got a bit tangled up! Let me try again."

    def generate_llm_response(self, text):
        """Generate a response using DeepSeek API."""
        # Only get conversation history if needed (reduce context size for speed)
        conversation_history = "" if len(self.memory.memory) <= 1 else self.memory.get_recent_context()

        # Construct the prompt
        prompt = f"""{self.personality_prompt}

        {conversation_history}
        Human: {text}
        Assistant:"""

        # Call DeepSeek API
        response = self.call_deepseek_api(prompt)

        # Additional processing to remove any self-introductions that might still appear
        response = self.remove_self_introductions(response)

        self.memory.add_exchange(text, response)
        return response

    def remove_self_introductions(self, response):
        """Remove any instances where the assistant introduces itself."""
        # Remove name prefix at the start
        if response.startswith("Tsun:"):
            response = response[len("Tsun:"):].strip()

        # Check other variations with a precompiled list
        name_prefixes = [
            "Tsun: ", "Tsun - ", "Tsun | ",
            "TSUN: ", "TSUN - ", "TSUN | ",
            "This is Tsun: ", "Tsun says: "
        ]

        for prefix in name_prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
                break  # Exit after first match

        # Additional introduction patterns
        intro_patterns = [
            "I'm Tsun, ", "I am Tsun, ", "This is Tsun, ",
            "It's Tsun, ", "It is Tsun, ", "Tsun here, ",
            "Hello, I'm Tsun", "Hi, I'm Tsun",
            "As a Mizutsune", "As your Mizutsune"
        ]

        for pattern in intro_patterns:
            if response.startswith(pattern):
                response = response[len(pattern):]
                # Capitalize first letter if needed
                if response and response[0].islower():
                    response = response[0].upper() + response[1:]
                break  # Exit after first match

        return response

    def setup_tts(self):
        """Set up text-to-speech engine and route output to VB-Cable."""
        self.speaker = tts.init()

        # Set the output device to VB-Cable Output
        voices = self.speaker.getProperty('voices')
        for voice in voices:
            if "CABLE Output" in voice.name:  # Use VB-Cable Output
                self.speaker.setProperty('voice', voice.id)
                print(f"Selected voice: {voice.name}")  # Debug print
                break

        # Adjust speech rate if needed
        self.speaker.setProperty("rate", 165)

    def speak(self, text):
        """Speak the text using TTS and route output to VB-Cable."""
        self.is_speaking = True
        temp_path = None

        try:
            # Get configurations or set defaults
            voice_index = getattr(self, 'voice_index', 0)
            speech_rate = getattr(self, 'speech_rate', 170)
            volume = getattr(self, 'volume', 1.0)
            semitones = getattr(self, 'semitones', 0)
            speed = getattr(self, 'speed', 1.0)

            # Create a temporary file for the audio
            with self.tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_path = temp_file.name

            # Initialize the TTS engine
            engine = tts.init()

            # Configure the engine to save to file
            engine.setProperty('volume', volume)
            engine.setProperty('rate', speech_rate)

            # Set voice to VB-Cable Output
            voices = engine.getProperty('voices')
            for voice in voices:
                if "CABLE Output" in voice.name:  # Use VB-Cable Output
                    engine.setProperty('voice', voice.id)
                    break

            # Save speech to the temporary WAV file
            engine.save_to_file(text, temp_path)
            engine.runAndWait()

            # Apply pitch and speed adjustments using pydub if needed
            if semitones != 0 or speed != 1.0:
                # Load the audio file
                sound = self.AudioSegment.from_file(temp_path, format="wav")

                # Apply pitch shift if needed
                if semitones != 0:
                    sound = sound._spawn(sound.raw_data, overrides={
                        "frame_rate": int(sound.frame_rate * (2.0 ** (semitones / 12.0)))
                    })
                    sound = sound.set_frame_rate(sound.frame_rate)

                # Apply speed adjustment if needed
                if speed != 1.0:
                    sound = sound.speedup(playback_speed=speed)

                # Play the modified audio
                self.play(sound)
            else:
                # Play the unmodified file directly
                sound = self.AudioSegment.from_file(temp_path, format="wav")
                self.play(sound)

        except Exception as e:
            print(f"Error in TTS: {e}")
            traceback.print_exc()
        finally:
            # Clean up temp file if it exists
            try:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass

        self.is_speaking = False
        time.sleep(0.2)  # Add a small delay after speaking

    def listen_for_audio(self, mic):
        """Listen for audio input - KEEPING ORIGINAL FUNCTIONALITY."""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Increase these values to allow for longer pauses between words
                self.recogniser.pause_threshold = 2.0  # Much longer pause allowed between words
                self.recogniser.phrase_threshold = 0.5  # Higher threshold for considering a phrase complete
                self.recogniser.non_speaking_duration = 1.0  # Longer silence needed to determine end of speech

                print("..." if attempt > 0 else "Listening...")

                # Shorter ambient noise adjustment
                self.recogniser.adjust_for_ambient_noise(mic, duration=0.3)

                # Critical parameters for fixing premature cutoff:
                audio = self.recogniser.listen(
                    mic,
                    timeout=10,  # Longer timeout to wait for speech to begin
                    phrase_time_limit=20  # Much longer maximum phrase time (was 10-15)
                )

                text = self.recogniser.recognize_google(audio, show_all=True)
                if isinstance(text, dict) and 'alternative' in text:
                    text = text['alternative'][0]['transcript']
                elif isinstance(text, list) and text:
                    text = text[0]

                if text:
                    return text.lower()

            except speech_recognition.WaitTimeoutError:
                print("No speech detected. Please try again.")
            except speech_recognition.UnknownValueError:
                print("Could not understand audio. Please try again.")
            except speech_recognition.RequestError as e:
                print(f"Could not request results; {e}")
            except Exception as e:
                print(f"Error: {str(e)}")

        return None

    def cleanup(self):
        """Clean shutdown of the application"""
        self.running = False
        sys.exit(0)

    def run_assistant(self):
        """Main loop for the assistant with improved microphone handling."""
        print("Assistant is running!")

        # Get available microphones once
        try:
            mics = speech_recognition.Microphone.list_microphone_names()
            print(f"Available microphones: {len(mics)}")
        except Exception as e:
            print(f"Error listing microphones: {e}")
            mics = []

        # Try multiple device indices
        device_indices_to_try = [0, 1, 2] if len(mics) > 2 else [0]

        # Common microphone configuration
        mic_config = {
            'sample_rate': 16000,
            'chunk_size': 1024
        }

        for device_index in device_indices_to_try:
            try:
                # Update mic config with current device index
                current_mic_config = mic_config.copy()
                current_mic_config['device_index'] = device_index

                print(f"Trying microphone with index {device_index}")

                with speech_recognition.Microphone(**current_mic_config) as mic:
                    # Adjust noise threshold
                    print("Adjusting for ambient noise...")
                    self.recogniser.adjust_for_ambient_noise(mic, duration=1)

                    # Print wake words
                    wake_word_list = '", "'.join(self.WAKE_WORDS)
                    print(f'Say "{wake_word_list}" to start...')

                    # Main listening loop
                    self.main_listening_loop(mic)

                # If we got here without exception, we found a working microphone
                break

            except Exception as e:
                print(f"Error with microphone {device_index}: {e}")
                traceback.print_exc()
                continue

        print("Could not initialize any microphone. Please check your audio setup.")

    def main_listening_loop(self, mic):
        """Main listening loop, extracted for clarity."""
        while self.running:
            try:
                if self.is_speaking:
                    time.sleep(0.1)
                    continue

                print("Listening...")
                audio = self.recogniser.listen(mic, timeout=5, phrase_time_limit=5)

                try:
                    text = self.recogniser.recognize_google(audio).lower()
                    print(f"Heard: {text}")

                    # Check for wake word using set for faster lookup
                    if any(word in text for word in self.WAKE_WORDS):
                        print("Wake word detected!")
                        initial_response = "Hey there!"
                        self.speak(initial_response)
                        self.memory.add_exchange(text, initial_response)

                        # Enter continuous command listening mode
                        self.continuous_command_listening(mic)

                except speech_recognition.UnknownValueError:
                    print("Could not understand audio.")
                except speech_recognition.RequestError as e:
                    print(f"Could not request results; {e}")

            except speech_recognition.WaitTimeoutError:
                print("No speech detected.")
            except Exception as e:
                print(f"Unexpected error in main listening loop: {e}")
                traceback.print_exc()
                time.sleep(1)  # Brief pause before continuing

            # Small sleep to prevent tight looping
            time.sleep(0.1)

    def continuous_command_listening(self, mic):
        """
        Continuous command listening mode that stays active until explicitly told to stop.
        """
        print("Entering continuous command listening mode.")

        # Adjust recognizer settings for command mode
        self.recogniser.pause_threshold = 2.0
        self.recogniser.phrase_threshold = 0.5
        self.recogniser.non_speaking_duration = 1.0

        while self.running:
            try:
                # Skip if currently speaking
                if self.is_speaking:
                    time.sleep(0.2)
                    continue

                print("Listening for command...")

                # Capture audio with a reasonable timeout
                try:
                    audio = self.recogniser.listen(mic, timeout=5, phrase_time_limit=10)
                except speech_recognition.WaitTimeoutError:
                    # If no speech is detected, continue listening
                    continue

                # Recognize speech
                try:
                    # First try Google Speech Recognition
                    text = self.recogniser.recognize_google(audio).lower()
                    print(f"Recognized command: {text}")

                    # Check for exit commands using set intersection for efficiency
                    if any(cmd in text for cmd in self.EXIT_COMMANDS):
                        farewell = "Bye bye!"
                        self.memory.add_exchange(text, farewell)
                        self.speak(farewell)
                        print("Exiting continuous command listening mode.")
                        break

                    # Check for configuration reload
                    if 'reload' in text or 'reconfigure' in text:
                        config_response = self.reload_config()
                        self.speak(config_response)
                        continue

                    # Generate response using LLM
                    response = self.get_llm_response(text)  # Only pass `text`
                    if response:
                        print(f"{response}")
                        self.speak(response)

                except speech_recognition.UnknownValueError:
                    print("Could not understand audio. Please try again.")
                except speech_recognition.RequestError as e:
                    print(f"Could not request results from speech recognition service; {e}")
                    self.speak("I'm having trouble understanding. Could you repeat that?")

            except Exception as e:
                # Catch-all for any unexpected errors
                print(f"Unexpected error in continuous command listening mode: {e}")
                traceback.print_exc()

                # Provide a recovery mechanism
                self.speak("I encountered an error. Let's continue listening.")
                time.sleep(1)  # Brief pause before continuing

            # Small sleep to prevent tight looping
            time.sleep(0.1)

        print("Continuous command listening mode ended.")
        return


if __name__ == "__main__":
    Assistant()
