// Status: in development.
// Step 11 validation test for websocket transport dispatch into BP_SampleManager.

#if WITH_DEV_AUTOMATION_TESTS

#include "../SampleManagerRuntimeActor.h"
#include "../SetDataConfigRuntimeActor.h"

#include "Dom/JsonObject.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Pawn.h"
#include "Misc/AutomationTest.h"
#include "Misc/Base64.h"

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
    FTransportDispatchStep11Test,
    "UE.Drone.ImageCapture.Step11.TransportDispatchToSampleManager",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FTransportDispatchStep11Test::RunTest(const FString& Parameters)
{
#if !WITH_EDITOR
    AddError(TEXT("Step11 test requires editor context."));
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

    const FString SetDataConfigLabel = TEXT("BP_SetDataConfig_Step11_Test");
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
    const FString DroneId = TEXT("drone_step11_0001");
    Pawn->Tags.Add(FName(*FString::Printf(TEXT("DroneId:%s"), *DroneId)));

    ASampleManagerRuntimeActor* SampleManager = nullptr;
    for (TActorIterator<ASampleManagerRuntimeActor> It(World); It; ++It)
    {
        ASampleManagerRuntimeActor* ExistingSampleManager = *It;
        if (ExistingSampleManager == nullptr)
        {
            continue;
        }
        if (SampleManager == nullptr)
        {
            SampleManager = ExistingSampleManager;
            continue;
        }
        ExistingSampleManager->Destroy();
    }

    if (SampleManager == nullptr)
    {
        SampleManager = World->SpawnActor<ASampleManagerRuntimeActor>(
            SampleManagerClass,
            FVector(600.0, 0.0, 120.0),
            FRotator::ZeroRotator,
            SpawnParameters
        );
    }
    if (!TestNotNull(TEXT("Resolved BP_SampleManager"), SampleManager))
    {
        Pawn->Destroy();
        SetDataConfigActor->Destroy();
        return false;
    }
    SampleManager->SetActorLabel(TEXT("BP_SampleManager_Main"));
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

    const FString RunId = TEXT("run_step11");
    const FString ConfigId = TEXT("config_step11");
    const FString ConfigHash = TEXT("hash_step11");
    const FString ConfigPayload = TEXT(
        "{\"sensor_rig\":{\"active_viewpoint\":\"front\",\"front_fov_deg\":76.0,\"capture_width\":320,\"capture_height\":200,"
        "\"front_offset_cm\":{\"x\":12.0,\"y\":0.0,\"z\":8.0},\"front_rotation_deg\":{\"pitch\":0.0,\"roll\":0.0,\"yaw\":0.0}}}"
    );
    const bool bAcceptedConfig = SetDataConfigActor->OnSetConfigReceived(RunId, ConfigId, ConfigHash, ConfigPayload);
    TestTrue(TEXT("Valid SET_CONFIG should be accepted"), bAcceptedConfig);
    TestTrue(TEXT("SetDataConfig actor should be config-ready"), SetDataConfigActor->bConfigReady);

    FString DirectObservationJson;
    FString DirectCaptureError;
    const bool bDirectCaptureSuccess = SampleManager->CaptureNow(
        RunId,
        DroneId,
        TEXT("cap_step11_direct"),
        DirectObservationJson,
        DirectCaptureError
    );
    TestTrue(TEXT("Direct SampleManager capture should succeed before transport dispatch"), bDirectCaptureSuccess);
    if (!bDirectCaptureSuccess)
    {
        AddError(FString::Printf(TEXT("Direct SampleManager CaptureNow failed: %s"), *DirectCaptureError));
        CleanupActors();
        return false;
    }

    const bool bDirectTransportCaptureSuccess = SampleManager->CaptureNowTransport(
        RunId,
        DroneId,
        TEXT("cap_step11_transport_direct")
    );
    TestTrue(TEXT("Direct SampleManager transport wrapper should succeed before dispatch"), bDirectTransportCaptureSuccess);
    if (!bDirectTransportCaptureSuccess)
    {
        AddError(
            FString::Printf(
                TEXT("Direct SampleManager CaptureNowTransport failed: %s"),
                *SampleManager->transport_last_error
            )
        );
        CleanupActors();
        return false;
    }
    TestTrue(
        TEXT("Direct transport wrapper should record non-empty observation JSON"),
        !SampleManager->transport_last_observation_json.IsEmpty()
    );

    FString ResolveError;
    AActor* ResolvedManager = SetDataConfigActor->ResolveSampleManagerActor(ResolveError);
    TestNotNull(TEXT("Transport actor should resolve BP_SampleManager"), ResolvedManager);
    if (ResolvedManager == nullptr)
    {
        AddError(FString::Printf(TEXT("ResolveSampleManagerActor failed: %s"), *ResolveError));
        CleanupActors();
        return false;
    }

    const FString CaptureId = TEXT("cap_step11_001");
    TSharedPtr<FJsonObject> ObservationPayload;
    FString ResolvedDroneId;
    FString ResolvedCaptureId;
    FString DispatchError;
    ASampleManagerRuntimeActor* TypedManager = Cast<ASampleManagerRuntimeActor>(ResolvedManager);
    if (!TestNotNull(TEXT("Resolved manager should be SampleManagerRuntimeActor"), TypedManager))
    {
        CleanupActors();
        return false;
    }

    const bool bCaptureDispatched = SetDataConfigActor->InvokeSampleManagerCaptureNow(
        TypedManager,
        RunId,
        DroneId,
        CaptureId,
        ObservationPayload,
        ResolvedDroneId,
        ResolvedCaptureId,
        DispatchError
    );
    TestTrue(TEXT("Transport dispatch capture should succeed"), bCaptureDispatched);
    if (!bCaptureDispatched)
    {
        AddError(FString::Printf(TEXT("InvokeSampleManagerCaptureNow failed: %s"), *DispatchError));
        CleanupActors();
        return false;
    }
    if (!TestTrue(TEXT("Observation payload should be valid"), ObservationPayload.IsValid()))
    {
        CleanupActors();
        return false;
    }

    TestEqual(TEXT("Resolved drone id should match request"), ResolvedDroneId, DroneId);
    TestEqual(TEXT("Resolved capture id should match request"), ResolvedCaptureId, CaptureId);

    const TSharedPtr<FJsonObject>* ConfigRefObject = nullptr;
    TestTrue(TEXT("Observation should include config_ref"), ObservationPayload->TryGetObjectField(TEXT("config_ref"), ConfigRefObject));
    if (ConfigRefObject != nullptr && ConfigRefObject->IsValid())
    {
        FString ObservedConfigId;
        FString ObservedConfigHash;
        TestTrue(TEXT("config_ref.config_id should exist"), (*ConfigRefObject)->TryGetStringField(TEXT("config_id"), ObservedConfigId));
        TestTrue(TEXT("config_ref.config_hash should exist"), (*ConfigRefObject)->TryGetStringField(TEXT("config_hash"), ObservedConfigHash));
        TestEqual(TEXT("config_ref.config_id should match"), ObservedConfigId, ConfigId);
        TestEqual(TEXT("config_ref.config_hash should match"), ObservedConfigHash, ConfigHash);
    }

    const TSharedPtr<FJsonObject>* ViewpointObject = nullptr;
    TestTrue(TEXT("Observation should include viewpoint"), ObservationPayload->TryGetObjectField(TEXT("viewpoint"), ViewpointObject));
    int32 ViewpointWidth = 0;
    int32 ViewpointHeight = 0;
    if (ViewpointObject != nullptr && ViewpointObject->IsValid())
    {
        TestTrue(TEXT("viewpoint.width should exist"), (*ViewpointObject)->TryGetNumberField(TEXT("width"), ViewpointWidth));
        TestTrue(TEXT("viewpoint.height should exist"), (*ViewpointObject)->TryGetNumberField(TEXT("height"), ViewpointHeight));
    }

    FString LegacyImageBytesBase64;
    TestTrue(TEXT("Observation should include image_bytes_b64"), ObservationPayload->TryGetStringField(TEXT("image_bytes_b64"), LegacyImageBytesBase64));
    TestTrue(TEXT("image_bytes_b64 should be non-empty"), !LegacyImageBytesBase64.IsEmpty());

    const TSharedPtr<FJsonObject>* ImageObject = nullptr;
    TestTrue(TEXT("Observation should include image object"), ObservationPayload->TryGetObjectField(TEXT("image"), ImageObject));
    if (ImageObject != nullptr && ImageObject->IsValid())
    {
        FString CanonicalImageBytesBase64;
        int32 CanonicalWidth = 0;
        int32 CanonicalHeight = 0;
        TestTrue(TEXT("image.bytes_b64 should exist"), (*ImageObject)->TryGetStringField(TEXT("bytes_b64"), CanonicalImageBytesBase64));
        TestTrue(TEXT("image.width should exist"), (*ImageObject)->TryGetNumberField(TEXT("width"), CanonicalWidth));
        TestTrue(TEXT("image.height should exist"), (*ImageObject)->TryGetNumberField(TEXT("height"), CanonicalHeight));
        TestEqual(TEXT("Canonical image bytes should match legacy bytes"), CanonicalImageBytesBase64, LegacyImageBytesBase64);
        TestEqual(TEXT("Canonical image width should match viewpoint.width"), CanonicalWidth, ViewpointWidth);
        TestEqual(TEXT("Canonical image height should match viewpoint.height"), CanonicalHeight, ViewpointHeight);
    }

    TArray<uint8> DecodedPngBytes;
    TestTrue(TEXT("Image base64 should decode"), FBase64::Decode(LegacyImageBytesBase64, DecodedPngBytes));
    int32 DecodedWidth = 0;
    int32 DecodedHeight = 0;
    TestTrue(TEXT("Decoded image should be PNG"), DecodePngDimensions(DecodedPngBytes, DecodedWidth, DecodedHeight));
    TestEqual(TEXT("Decoded image width should match viewpoint width"), DecodedWidth, ViewpointWidth);
    TestEqual(TEXT("Decoded image height should match viewpoint height"), DecodedHeight, ViewpointHeight);

    CleanupActors();
    return !HasAnyErrors();
#endif
}

#endif
