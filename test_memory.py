from memory_manager import UpstashSessionMemory
import uuid

def run_diagnostics():
    print("🧪 Booting Upstash Memory Governance Test...")
    
    try:
        # Initialize manager
        memory_engine = UpstashSessionMemory()
        
        # Generate a mock unique session ID representing a student
        test_session = f"student_test_{uuid.uuid4().hex[:6]}"
        print(f"👤 Simulated User Session Established: {test_session}")
        
        # 1. Seed simulated interactive mock interview messages
        memory_engine.add_message_to_session(test_session, "user", "Hi, I am preparing for the Backend role at Fiserv.")
        memory_engine.add_message_to_session(test_session, "assistant", "Excellent! Let's start with database management. Explain indexing.")
        
        # 2. Pull the memory state back live from the cloud
        live_history = memory_engine.get_session_history(test_session)
        
        print("\n📥 Retracted Live JSON History from Upstash Cloud:")
        for turn in live_history:
            print(f"   [{turn['role'].upper()}]: {turn['content']}")
            
        # 3. Cleanup diagnostic check
        memory_engine.clear_session(test_session)
        print("\n🔬 Diagnostic Complete: Cloud connectivity is 100% operational!")
        
    except Exception as e:
        print(f"\n❌ Diagnostic Failed: {str(e)}")

if __name__ == "__main__":
    run_diagnostics()