// Status: in development.
// Step 9C validation test for config-reference handoff ownership and behavior.

#if WITH_DEV_AUTOMATION_TESTS

#include "../SetDataConfigRuntimeActor.h"

#include "Engine/World.h"
#include "GameFramework/Pawn.h"
#include "Misc/AutomationTest.h"

#if WITH_EDITOR
#include "Editor.h"
#endif

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FConfigReferenceStep9CTest,
    "UE.Drone.ImageCapture.Step9C.ConfigReferenceHandoff",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FConfigReferenceStep9CTest::RunTest(const FString& Parameters)
{
#if !WITH_EDITOR
    AddError(TEXT("Step9C test requires editor context."));
    return false;
#else
    UWorld* World = nullptr;
    if (GEditor != nullptr)
    {
        World = GEditor->GetEditorWorldContext().World();
    }
    if (!TestNotNull(TEXT("Editor world"), World))
    {
        return false;
    }

    UClass* SetDataConfigClass = StaticLoadClass(
        ASetDataConfigRuntimeActor::StaticClass(),
        nullptr,
        TEXT("/Game/io/Data_Config/BP_SetDataConfig.BP_SetDataConfig_C")
    );
    if (!TestNotNull(TEXT("BP_SetDataConfig class"), SetDataConfigClass))
    {
        return false;
    }

    UClass* PawnClass = StaticLoadClass(
        APawn::StaticClass(),
        nullptr,
        TEXT("/Game/Drone_Content/Blueprints/BP_DronePawn.BP_DronePawn_C")
    );
    if (!TestNotNull(TEXT("BP_DronePawn class"), PawnClass))
    {
        return false;
    }

    FActorSpawnParameters SpawnParameters;
    SpawnParameters.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

    ASetDataConfigRuntimeActor* SetDataConfigActor = World->SpawnActor<ASetDataConfigRuntimeActor>(
        SetDataConfigClass,
        FVector(0.0, 0.0, 120.0),
        FRotator::ZeroRotator,
        SpawnParameters
    );
    if (!TestNotNull(TEXT("Spawned BP_SetDataConfig"), SetDataConfigActor))
    {
        return false;
    }

    APawn* Pawn = World->SpawnActor<APawn>(
        PawnClass,
        FVector(300.0, 0.0, 120.0),
        FRotator::ZeroRotator,
        SpawnParameters
    );
    if (!TestNotNull(TEXT("Spawned BP_DronePawn"), Pawn))
    {
        SetDataConfigActor->Destroy();
        return false;
    }

    const auto CleanupActors = [&]()
    {
        if (IsValid(Pawn))
        {
            Pawn->Destroy();
        }
        if (IsValid(SetDataConfigActor))
        {
            SetDataConfigActor->Destroy();
        }
    };

    const auto BuildPayloadJson = [](int32 CaptureWidth) -> FString
    {
        return FString::Printf(
            TEXT("{\"sensor_rig\":{\"active_viewpoint\":\"front\",\"front_fov_deg\":77.5,\"capture_width\":%d,\"capture_height\":200,\"front_offset_cm\":{\"x\":15.0,\"y\":0.0,\"z\":8.0},\"front_rotation_deg\":{\"pitch\":0.0,\"roll\":0.0,\"yaw\":0.0}}}"),
            CaptureWidth
        );
    };

    const FString ValidRunId = TEXT("run_step9c_valid");
    const FString ValidConfigId = TEXT("config_step9c_valid");
    const FString ValidConfigHash = TEXT("hash_step9c_valid");
    const bool bAcceptedValid = SetDataConfigActor->OnSetConfigReceived(
        ValidRunId,
        ValidConfigId,
        ValidConfigHash,
        BuildPayloadJson(320)
    );
    TestTrue(TEXT("Valid SET_CONFIG should be accepted"), bAcceptedValid);
    TestTrue(TEXT("bConfigReady should be true after valid SET_CONFIG"), SetDataConfigActor->bConfigReady);

    FString ActiveConfigId;
    FString ActiveConfigHash;
    const bool bHasActiveConfigRef = SetDataConfigActor->GetActiveConfigReference(ActiveConfigId, ActiveConfigHash);
    TestTrue(TEXT("Active config reference should be readable after valid SET_CONFIG"), bHasActiveConfigRef);
    TestEqual(TEXT("Active config_id should match accepted config"), ActiveConfigId, ValidConfigId);
    TestEqual(TEXT("Active config_hash should match accepted config"), ActiveConfigHash, ValidConfigHash);

    const FString InvalidRunId = TEXT("run_step9c_invalid");
    const FString InvalidConfigId = TEXT("config_step9c_invalid");
    const FString InvalidConfigHash = TEXT("hash_step9c_invalid");
    const bool bAcceptedInvalid = SetDataConfigActor->OnSetConfigReceived(
        InvalidRunId,
        InvalidConfigId,
        InvalidConfigHash,
        BuildPayloadJson(0)
    );
    TestFalse(TEXT("Invalid SET_CONFIG should be rejected"), bAcceptedInvalid);
    TestFalse(TEXT("bConfigReady should be false after rejected SET_CONFIG"), SetDataConfigActor->bConfigReady);
    TestTrue(TEXT("LastError should be populated after rejected SET_CONFIG"), !SetDataConfigActor->LastError.IsEmpty());

    FString ActiveConfigIdAfterReject;
    FString ActiveConfigHashAfterReject;
    const bool bHasConfigRefAfterReject = SetDataConfigActor->GetActiveConfigReference(
        ActiveConfigIdAfterReject,
        ActiveConfigHashAfterReject
    );
    TestFalse(TEXT("Active config reference should be unavailable after rejected SET_CONFIG"), bHasConfigRefAfterReject);
    TestTrue(TEXT("config_id should be cleared after rejected SET_CONFIG"), ActiveConfigIdAfterReject.IsEmpty());
    TestTrue(TEXT("config_hash should be cleared after rejected SET_CONFIG"), ActiveConfigHashAfterReject.IsEmpty());

    CleanupActors();
    return !HasAnyErrors();
#endif
}

#endif
