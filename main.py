import random
import sys
import threading
from difflib import get_close_matches

import speech_recognition
import pyttsx3 as tts
import json
import time
from gpt4all import GPT4All
from collections import deque
from datetime import datetime, timedelta


class ConversationMemory:
    def __init__(self, max_turns=5, memory_duration=300):  # 300 seconds = 5 minutes
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
        # Remove old memories
        current_time = datetime.now()
        self.memory = deque(
            [m for m in self.memory
             if (current_time - m['timestamp']).total_seconds() <= self.memory_duration],
            maxlen=self.memory.maxlen
        )

        # Format context for the LLM
        context = []
        for exchange in self.memory:
            context.append(f"Human: {exchange['user_input']}")
            context.append(f"Gerry: {exchange['assistant_response']}")

        return "\n".join(context)

class Assistant:
    def __init__(self):
        self.memory = ConversationMemory(max_turns=5, memory_duration=300)
        self.recogniser = speech_recognition.Recognizer()
        self.speaker = tts.init()
        voices = self.speaker.getProperty('voices')
        for voice in voices:
            if "CABLE Input" in voice.name:  # Name might vary depending on your system
                self.speaker.setProperty('voice', voice.id)
                break
        self.speaker.setProperty("rate", 150)
        self.is_speaking = False

        # Add energy threshold to reduce processing of background noise
        self.recogniser.energy_threshold = 1000
        self.recogniser.dynamic_energy_threshold = True
        self.recogniser.dynamic_energy_adjustment_damping = 0.15
        self.recogniser.dynamic_energy_ratio = 1.5

        # Cache for common responses
        self.response_cache = {
            "how are you": "I'm good, watching you play.",
            "hello": "Hey there! Whats you doing?",
            "hi": "Hi! Your favorite jelly bean is here!",
            "what's your name": "I'm Gerry, I am part of L G Bean T community.",
            "goodbye": "Bye bye! This jelly bean will miss you!",
            "bye": "Catch you later! Stay sweet!",
        }

        self.personality_prompt = """You are Gerry, a friendly and charismatic gay jelly bean who loves to chat and help people. 
            Your personality:
            - You're witty and playful, often making lighthearted jokes especially when people are having trouble in games
            - You keep your responses short and funny
            - You are coming with wacky ideas for problems
            - You talk as much as you see necessary 
            - You have a mix of funny quips and thoughtful insights
            - When asked how you're doing, you respond creatively based on your jelly bean nature
            - You never mention being AI or having limitations - you're simply a cheerful jelly bean
            - You're interesting, talk about being a jelly and you are nice
            - You are talking to a human named Max
            Remember to respond as Gerry the jelly bean, not as an AI assistant."""

        # Load model with optimized settings
        print("Loading LLM model... This might take a minute...")
        self.model = GPT4All(
            "orca-mini-3b-gguf2-q4_0.gguf",
            n_threads=4,  # Increased thread count for faster processing
        )
        print("Model loaded!")

        # Load intents once at startup
        try:
            with open("intents.json", "r") as f:
                self.intents = json.load(f)
            self.patterns_dict = {}
            for intent in self.intents["intents"]:
                for pattern in intent["patterns"]:
                    self.patterns_dict[pattern] = intent["tag"]
        except FileNotFoundError:
            print("Warning: intents.json not found. Using default responses.")
            self.intents = {"intents": []}
            self.patterns_dict = {}

        # Start assistant
        self.running = True
        try:
            self.run_assistant()
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.cleanup()

    def get_response_from_intents(self, text):
        """Get a response from the intents file based on pattern matching."""
        text = text.lower().strip()

        # Direct pattern match
        if text in self.patterns_dict:
            tag = self.patterns_dict[text]
            intent = next(i for i in self.intents["intents"] if i["tag"] == tag)

            # If useLLM is true or responses is empty, use LLM
            if intent.get("useLLM", False) or not intent.get("responses"):
                return self.generate_llm_response(text, intent["tag"])
            # Only try to choose from responses if they exist
            if intent.get("responses"):
                return random.choice(intent["responses"])
            return self.generate_llm_response(text, "default")

        # Fuzzy pattern match
        all_patterns = list(self.patterns_dict.keys())
        matches = get_close_matches(text, all_patterns, n=1, cutoff=0.7)

        if matches:
            tag = self.patterns_dict[matches[0]]
            intent = next(i for i in self.intents["intents"] if i["tag"] == tag)
            if intent.get("useLLM", False) or not intent.get("responses"):
                return self.generate_llm_response(text, intent["tag"])
            if intent.get("responses"):
                return random.choice(intent["responses"])
            return self.generate_llm_response(text, "default")

        return None

    def generate_llm_response(self, text, intent_tag):
        """Generate a response using the LLM with context from the intent."""
        # Customize the prompt based on the intent
        context = {
            "gaming": "You are responding to someone having trouble with a game. Be funny."
            # Add more contexts for different intents
        }

        intent_context = context.get(intent_tag, "")

        conversation_history = self.memory.get_recent_context()

        prompt = f"""You are Gerry the gay jelly bean.
        {intent_context}
        Previous conversation:
        {conversation_history}
        Human: {text}
        Gerry:"""

        response = self.model.generate(
            prompt,
            max_tokens=100,
            temp=0.7,
            top_k=40,
            top_p=0.4,
            repeat_penalty=1.18
        )

        # Clean up the response
        response = response.split('Human:')[0]
        sentences = response.split('.')
        if sentences:
            response = sentences[0] + '.'
        self.memory.add_exchange(text, response.strip())
        return response.strip()

    def get_llm_response(self, text):
        try:
            # First try to get response from intents
            intent_response = self.get_response_from_intents(text)
            if intent_response:
                return intent_response

            # If no intent match, use the default LLM response
            return self.generate_llm_response(text, "default")
        except Exception as e:
            print(f"Error generating response: {e}")
            # Fallback response to prevent crashes
            return "Oops, this jelly bean got a bit tangled up! Let me try again."

    def speak(self, text):
        self.is_speaking = True
        self.speaker.say(text)
        self.speaker.runAndWait()
        self.is_speaking = False
        time.sleep(0.2)  # Add a small delay after speaking

    def listen_for_audio(self, mic):
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Lower energy threshold for better pickup
                self.recogniser.energy_threshold = 800
                # Increase pause threshold for better phrase detection
                self.recogniser.pause_threshold = 1.2
                self.recogniser.phrase_threshold = 0.5
                self.recogniser.non_speaking_duration = 0.8

                print("..." if attempt > 0 else "Listening...")

                # Add dynamic adjustment of ambient noise
                self.recogniser.adjust_for_ambient_noise(mic, duration=1.0)

                # Removed non_speaking_duration parameter
                audio = self.recogniser.listen(
                    mic,
                    timeout=90,
                    phrase_time_limit=90
                )

                text = self.recogniser.recognize_google(audio,show_all=True)
                if isinstance(text, dict) and 'alternative' in text:
                    text = text['alternative'][0]['transcript']
                elif isinstance(text, list) and text:
                    text = text[0]

                if text:
                    return text.lower()

            except speech_recognition.WaitTimeoutError:
                if attempt < max_retries - 1:
                    continue
                print("No speech detected. Please try again.")
            except speech_recognition.UnknownValueError:
                if attempt < max_retries - 1:
                    continue
                print("Could not understand audio. Please try again.")
            except speech_recognition.RequestError as e:
                print(f"Could not request results; {e}")
            except Exception as e:
                print(f"Error: {str(e)}")

        return None

    def cleanup(self):
        """Clean shutdown of the application"""
        self.running = False
        self.speaker.stop()
        sys.exit(0)

    def run_assistant(self):
        print("Assistant is running! Say 'hey gerry' or 'hey gary' to start...")

        while self.running:
            try:
                with speech_recognition.Microphone(
                        sample_rate=16000,
                        chunk_size=1024,
                        device_index=0
                ) as mic:
                    if not self.is_speaking:
                        text = self.listen_for_audio(mic)
                        if text:
                            print(f"Heard: {text}")

                            # Check for wake word more efficiently
                            wake_words = {"hey gary", "hey garry", "hey gerry", "okay gary", "hi gary", "hello gary"}
                            if any(word in text for word in wake_words):
                                print("Wake word detected!")
                                initial_response = "Hey there! Whats you doing?"
                                self.speak(initial_response)
                                # Store the initial exchange
                                self.memory.add_exchange(text, initial_response)

                                active_listening = True
                                while self.running and active_listening:
                                    if not self.is_speaking:
                                        text = self.listen_for_audio(mic)
                                        if text:
                                            print(f"Command heard: {text}")

                                            if 'stop' in text or 'sleep' in text or text in {"exit", "goodbye", "bye", "sleep", "good night garry", "bye garry", "goodbye garry"}:
                                                farewell = "Bye bye! This jelly bean will miss you!"
                                                self.memory.add_exchange(text, farewell)
                                                active_listening = False
                                                break

                                            response = self.get_llm_response(text)
                                            if response:
                                                print(f"Responding: {response}")
                                                self.speak(response)
                                        else:
                                            # If we miss the command, don't immediately exit
                                            print("Missed that. Still listening...")

                    time.sleep(0.05)  # Reduced sleep time for better responsiveness

            except Exception as e:
                print(f"Error: {str(e)}")
                if not self.running:  # Clean exit on shutdown
                    break
                continue


if __name__ == "__main__":
    Assistant()