// Status: in development.
// Step 9D validation test for BP_SampleManager native orchestration behavior.

#if WITH_DEV_AUTOMATION_TESTS

#include "../SampleManagerRuntimeActor.h"
#include "../SetDataConfigRuntimeActor.h"

#include "Dom/JsonObject.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Pawn.h"
#include "Math/UnrealMathUtility.h"
#include "Misc/AutomationTest.h"
#include "Misc/Base64.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

#if WITH_EDITOR
#include "Editor.h"
#endif

namespace
{
bool DecodePngDimensions(const TArray<uint8>& PngBytes, int32& OutWidth, int32& OutHeight)
{
    OutWidth = 0;
    OutHeight = 0;
    if (PngBytes.Num() < 24)
    {
        return false;
    }

    static const uint8 PngSignature[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    for (int32 Index = 0; Index < 8; ++Index)
    {
        if (PngBytes[Index] != PngSignature[Index])
        {
            return false;
        }
    }

    OutWidth = (static_cast<int32>(PngBytes[16]) << 24)
        | (static_cast<int32>(PngBytes[17]) << 16)
        | (static_cast<int32>(PngBytes[18]) << 8)
        | static_cast<int32>(PngBytes[19]);
    OutHeight = (static_cast<int32>(PngBytes[20]) << 24)
        | (static_cast<int32>(PngBytes[21]) << 16)
        | (static_cast<int32>(PngBytes[22]) << 8)
        | static_cast<int32>(PngBytes[23]);
    return OutWidth > 0 && OutHeight > 0;
}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSampleManagerRuntimeStep9DTest,
    "UE.Drone.ImageCapture.Step9D.SampleManagerAtomicOrchestration",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FSampleManagerRuntimeStep9DTest::RunTest(const FString& Parameters)
{
#if !WITH_EDITOR
    AddError(TEXT("Step9D test requires editor context."));
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

    UClass* SampleManagerClass = StaticLoadClass(
        ASampleManagerRuntimeActor::StaticClass(),
        nullptr,
        TEXT("/Game/io/Data_Interface/BP_SampleManager.BP_SampleManager_C")
    );
    if (!TestNotNull(TEXT("BP_SampleManager class"), SampleManagerClass))
    {
        return false;
    }

    FActorSpawnParameters SpawnParameters;
    SpawnParameters.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

    for (TActorIterator<APawn> It(World); It; ++It)
    {
        APawn* ExistingPawn = *It;
        if (ExistingPawn != nullptr)
        {
            ExistingPawn->Destroy();
        }
    }

    const FString SetDataConfigLabel = TEXT("BP_SetDataConfig_Step9D_Test");
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
    SetDataConfigActor->SetActorLabel(SetDataConfigLabel);

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
    const FString DroneId = TEXT("drone_step9d_0001");
    Pawn->Tags.Add(FName(*FString::Printf(TEXT("DroneId:%s"), *DroneId)));

    ASampleManagerRuntimeActor* SampleManager = World->SpawnActor<ASampleManagerRuntimeActor>(
        SampleManagerClass,
        FVector(600.0, 0.0, 120.0),
        FRotator::ZeroRotator,
        SpawnParameters
    );
    if (!TestNotNull(TEXT("Spawned BP_SampleManager"), SampleManager))
    {
        Pawn->Destroy();
        SetDataConfigActor->Destroy();
        return false;
    }
    SampleManager->SetActorLabel(TEXT("BP_SampleManager_Step9D_Test"));
    SampleManager->config_reference_source_actor_label = SetDataConfigLabel;
    SampleManager->config_reference_read_function = TEXT("GetActiveConfigReference");

    const auto CleanupActors = [&]()
    {
        if (IsValid(SampleManager))
        {
            SampleManager->Destroy();
        }
        if (IsValid(Pawn))
        {
            Pawn->Destroy();
        }
        if (IsValid(SetDataConfigActor))
        {
            SetDataConfigActor->Destroy();
        }
    };

    FString ObservationJsonBeforeConfig;
    FString CaptureErrorBeforeConfig;
    const bool bCaptureBeforeConfig = SampleManager->CaptureNow(
        TEXT("run_step9d"),
        DroneId,
        TEXT("cap_before_config"),
        ObservationJsonBeforeConfig,
        CaptureErrorBeforeConfig
    );
    TestFalse(TEXT("Capture should fail before config is applied"), bCaptureBeforeConfig);
    TestTrue(TEXT("Capture-before-config error should be populated"), !CaptureErrorBeforeConfig.IsEmpty());

    const FString RunId = TEXT("run_step9d");
    const FString ConfigId = TEXT("config_step9d");
    const FString ConfigHash = TEXT("hash_step9d");
    const FString ConfigPayload = TEXT(
        "{\"sensor_rig\":{\"active_viewpoint\":\"front\",\"front_fov_deg\":77.5,\"capture_width\":320,\"capture_height\":200,"
        "\"front_offset_cm\":{\"x\":15.0,\"y\":0.0,\"z\":8.0},\"front_rotation_deg\":{\"pitch\":0.0,\"roll\":0.0,\"yaw\":0.0}}}"
    );
    const bool bAcceptedConfig = SetDataConfigActor->OnSetConfigReceived(RunId, ConfigId, ConfigHash, ConfigPayload);
    TestTrue(TEXT("Valid SET_CONFIG should be accepted"), bAcceptedConfig);
    TestTrue(TEXT("bConfigReady should be true after valid SET_CONFIG"), SetDataConfigActor->bConfigReady);

    const FString CaptureId = TEXT("cap_step9d_001");
    FString ObservationJson;
    FString CaptureError;
    const bool bCaptureSuccess = SampleManager->CaptureNow(RunId, DroneId, CaptureId, ObservationJson, CaptureError);
    TestTrue(TEXT("Capture should succeed after config apply"), bCaptureSuccess);
    TestTrue(TEXT("Capture success should return non-empty observation JSON"), !ObservationJson.IsEmpty());
    if (!bCaptureSuccess)
    {
        AddError(FString::Printf(TEXT("CaptureNow failed: %s"), *CaptureError));
        CleanupActors();
        return false;
    }

    TSharedPtr<FJsonObject> ObservationObject;
    const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(ObservationJson);
    TestTrue(TEXT("Observation JSON should deserialize"), FJsonSerializer::Deserialize(Reader, ObservationObject));
    if (!TestTrue(TEXT("Observation object should be valid"), ObservationObject.IsValid()))
    {
        CleanupActors();
        return false;
    }

    const TSharedPtr<FJsonObject>* ConfigRefObject = nullptr;
    TestTrue(TEXT("config_ref object should exist"), ObservationObject->TryGetObjectField(TEXT("config_ref"), ConfigRefObject));
    if (ConfigRefObject != nullptr && ConfigRefObject->IsValid())
    {
        FString ObservedConfigId;
        FString ObservedConfigHash;
        TestTrue(TEXT("config_ref.config_id should exist"), (*ConfigRefObject)->TryGetStringField(TEXT("config_id"), ObservedConfigId));
        TestTrue(TEXT("config_ref.config_hash should exist"), (*ConfigRefObject)->TryGetStringField(TEXT("config_hash"), ObservedConfigHash));
        TestEqual(TEXT("config_ref.config_id should match accepted config"), ObservedConfigId, ConfigId);
        TestEqual(TEXT("config_ref.config_hash should match accepted config"), ObservedConfigHash, ConfigHash);
    }

    const TSharedPtr<FJsonObject>* ViewpointObject = nullptr;
    TestTrue(TEXT("viewpoint object should exist"), ObservationObject->TryGetObjectField(TEXT("viewpoint"), ViewpointObject));
    int32 ViewpointWidth = 0;
    int32 ViewpointHeight = 0;
    double ViewpointFov = 0.0;
    if (ViewpointObject != nullptr && ViewpointObject->IsValid())
    {
        TestTrue(TEXT("viewpoint.width should exist"), (*ViewpointObject)->TryGetNumberField(TEXT("width"), ViewpointWidth));
        TestTrue(TEXT("viewpoint.height should exist"), (*ViewpointObject)->TryGetNumberField(TEXT("height"), ViewpointHeight));
        TestTrue(TEXT("viewpoint.fov_deg should exist"), (*ViewpointObject)->TryGetNumberField(TEXT("fov_deg"), ViewpointFov));

        FString ViewpointRunId;
        FString ViewpointCaptureId;
        TestTrue(TEXT("viewpoint.run_id should exist"), (*ViewpointObject)->TryGetStringField(TEXT("run_id"), ViewpointRunId));
        TestTrue(TEXT("viewpoint.capture_id should exist"), (*ViewpointObject)->TryGetStringField(TEXT("capture_id"), ViewpointCaptureId));
        TestEqual(TEXT("viewpoint.run_id should match"), ViewpointRunId, RunId);
        TestEqual(TEXT("viewpoint.capture_id should match"), ViewpointCaptureId, CaptureId);
        TestEqual(TEXT("viewpoint width should match config"), ViewpointWidth, 320);
        TestEqual(TEXT("viewpoint height should match config"), ViewpointHeight, 200);
        TestTrue(TEXT("viewpoint fov should match config"), FMath::IsNearlyEqual(ViewpointFov, 77.5, 0.01));
    }

    FString LegacyImageBytesBase64;
    TestTrue(TEXT("legacy image_bytes_b64 should exist"), ObservationObject->TryGetStringField(TEXT("image_bytes_b64"), LegacyImageBytesBase64));
    TestTrue(TEXT("legacy image_bytes_b64 should be non-empty"), !LegacyImageBytesBase64.IsEmpty());

    const TSharedPtr<FJsonObject>* ImageObject = nullptr;
    TestTrue(TEXT("canonical image object should exist"), ObservationObject->TryGetObjectField(TEXT("image"), ImageObject));
    if (ImageObject != nullptr && ImageObject->IsValid())
    {
        FString CanonicalImageBytesBase64;
        int32 CanonicalWidth = 0;
        int32 CanonicalHeight = 0;
        TestTrue(TEXT("image.bytes_b64 should exist"), (*ImageObject)->TryGetStringField(TEXT("bytes_b64"), CanonicalImageBytesBase64));
        TestTrue(TEXT("image.width should exist"), (*ImageObject)->TryGetNumberField(TEXT("width"), CanonicalWidth));
        TestTrue(TEXT("image.height should exist"), (*ImageObject)->TryGetNumberField(TEXT("height"), CanonicalHeight));
        TestEqual(TEXT("image.bytes_b64 should match legacy image_bytes_b64"), CanonicalImageBytesBase64, LegacyImageBytesBase64);
        TestEqual(TEXT("image.width should match viewpoint.width"), CanonicalWidth, ViewpointWidth);
        TestEqual(TEXT("image.height should match viewpoint.height"), CanonicalHeight, ViewpointHeight);
    }

    TArray<uint8> DecodedPngBytes;
    TestTrue(TEXT("PNG base64 decode should succeed"), FBase64::Decode(LegacyImageBytesBase64, DecodedPngBytes));
    int32 DecodedWidth = 0;
    int32 DecodedHeight = 0;
    TestTrue(TEXT("Decoded bytes should be a PNG"), DecodePngDimensions(DecodedPngBytes, DecodedWidth, DecodedHeight));
    TestEqual(TEXT("Decoded PNG width should match viewpoint.width"), DecodedWidth, ViewpointWidth);
    TestEqual(TEXT("Decoded PNG height should match viewpoint.height"), DecodedHeight, ViewpointHeight);

    CleanupActors();
    return !HasAnyErrors();
#endif
}

#endif
