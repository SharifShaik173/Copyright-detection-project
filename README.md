

🔐 Homomorphic Encryption-Based Secure Video Storage in Cloud

📌 What is this project?

This project stores multiple videos in the cloud using **Homomorphic Encryption (HE)** — a special encryption technique that allows the cloud server to process and manage your videos **without ever decrypting them**. Your data stays encrypted from upload to download, even during computation.


 
 ❌ Existing System — Before

Traditional cloud video storage **must decrypt data on the server** to process it. This creates serious risks:

- 🔓 Cloud provider can see your raw video content
- 💥 Server breach directly exposes all private videos
- 👁️ No privacy during processing — only during transmission
- 🕵️ Vulnerable to insider attacks and snooping
- 📤 Data leaves encryption boundary during computation
- 🚨 One hack = complete data leak


 
 
 ✅ Proposed System — Now

With Homomorphic Encryption, the cloud **never sees your actual video** at any stage:

- 🔐 Videos encrypted on client side before upload
- ☁️ Cloud stores and processes only ciphertext — never plaintext
- 🔍 Search and queries work directly on encrypted data
- 🗝️ Only the user with the private key can decrypt and view
- 🛡️ Server breach reveals nothing — only meaningless ciphertext
- 🔒 End-to-end privacy guaranteed at all times


 
 
 ⚖️ Comparison

| | ❌ Existing System | ✅ Proposed System |

| 👁️ Data visible to cloud | Yes | No |
| 🔓 Decrypted during processing | Yes | Never |
| 💥 Risk on server breach | High | Zero |
| 🛡️ User privacy | Partial | Complete |
| 🔐 Encrypted computation | No | Yes |
| 🗝️ Key stays with user | No | Always |
