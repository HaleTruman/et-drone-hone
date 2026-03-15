// Desired behavior:
// - Register a minimal runtime plugin module for DroneWebSocket.
// - Keep startup/shutdown lightweight with no hidden side effects.
// - Provide a stable module anchor for Blueprint-facing WebSocket component code.

#include "Modules/ModuleManager.h"

class FDroneWebSocketModule final : public IModuleInterface
{
public:
    virtual void StartupModule() override {}
    virtual void ShutdownModule() override {}
};

IMPLEMENT_MODULE(FDroneWebSocketModule, DroneWebSocket)
