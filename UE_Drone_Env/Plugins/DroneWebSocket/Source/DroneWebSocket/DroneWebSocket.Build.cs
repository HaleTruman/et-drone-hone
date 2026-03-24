// Desired behavior:
// - Define minimal runtime module dependencies for the DroneWebSocket plugin.
// - Keep module small and focused on Blueprint-facing WebSocket transport only.
// - Enable compilation in UE Editor/Runtime without introducing non-essential dependencies.

using UnrealBuildTool;

public class DroneWebSocket : ModuleRules
{
    public DroneWebSocket(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(
            new[]
            {
                "Core",
                "CoreUObject",
                "Engine",
            }
        );

        PrivateDependencyModuleNames.AddRange(
            new[]
            {
                "WebSockets",
                "Json",
                "JsonUtilities",
            }
        );
    }
}
