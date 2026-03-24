// Status: in development.
// Step 9B validation test for BP_DroneSensors native callable behavior.

#if WITH_DEV_AUTOMATION_TESTS

#include "../DroneSensorsRuntimeComponent.h"

#include "Components/ActorComponent.h"
#include "Engine/World.h"
#include "GameFramework/Pawn.h"
#include "Misc/AutomationTest.h"
#include "UObject/UnrealType.h"

#if WITH_EDITOR
#include "Editor.h"
#endif

namespace
{
const FProperty* FindPropertyByStableName(const UClass* Class, const TCHAR* PropertyName)
{
    if (Class == nullptr || PropertyName == nullptr || !*PropertyName)
    {
        return nullptr;
    }

    if (const FProperty* Exact = Class->FindPropertyByName(FName(PropertyName)))
    {
        return Exact;
    }

    const FString Wanted(PropertyName);
    const FString WantedPrefix = Wanted + TEXT("_");
    for (TFieldIterator<FProperty> It(Class, EFieldIterationFlags::IncludeSuper); It; ++It)
    {
        const FProperty* Candidate = *It;
        if (Candidate == nullptr)
        {
            continue;
        }

        const FString CandidateName = Candidate->GetName();
        if (CandidateName.Equals(Wanted, ESearchCase::IgnoreCase)
            || CandidateName.StartsWith(WantedPrefix, ESearchCase::IgnoreCase))
        {
            return Candidate;
        }
    }

    return nullptr;
}

FProperty* FindPropertyByStableName(UClass* Class, const TCHAR* PropertyName)
{
    return const_cast<FProperty*>(FindPropertyByStableName(static_cast<const UClass*>(Class), PropertyName));
}

bool SetNameLikeProperty(UObject* Target, const TCHAR* PropertyName, const FName& Value)
{
    if (Target == nullptr)
    {
        return false;
    }

    FProperty* Property = FindPropertyByStableName(Target->GetClass(), PropertyName);
    if (Property == nullptr)
    {
        return false;
    }

    if (FNameProperty* NameProperty = CastField<FNameProperty>(Property))
    {
        NameProperty->SetPropertyValue_InContainer(Target, Value);
        return true;
    }
    if (FStrProperty* StringProperty = CastField<FStrProperty>(Property))
    {
        StringProperty->SetPropertyValue_InContainer(Target, Value.ToString());
        return true;
    }
    return false;
}

bool SetRealLikeProperty(UObject* Target, const TCHAR* PropertyName, double Value)
{
    if (Target == nullptr)
    {
        return false;
    }

    FProperty* Property = FindPropertyByStableName(Target->GetClass(), PropertyName);
    if (Property == nullptr)
    {
        return false;
    }

    if (FFloatProperty* FloatProperty = CastField<FFloatProperty>(Property))
    {
        FloatProperty->SetPropertyValue_InContainer(Target, static_cast<float>(Value));
        return true;
    }
    if (FDoubleProperty* DoubleProperty = CastField<FDoubleProperty>(Property))
    {
        DoubleProperty->SetPropertyValue_InContainer(Target, Value);
        return true;
    }
    return false;
}

bool SetIntLikeProperty(UObject* Target, const TCHAR* PropertyName, int32 Value)
{
    if (Target == nullptr)
    {
        return false;
    }

    FProperty* Property = FindPropertyByStableName(Target->GetClass(), PropertyName);
    if (Property == nullptr)
    {
        return false;
    }

    if (FIntProperty* IntProperty = CastField<FIntProperty>(Property))
    {
        IntProperty->SetPropertyValue_InContainer(Target, Value);
        return true;
    }
    if (FInt64Property* Int64Property = CastField<FInt64Property>(Property))
    {
        Int64Property->SetPropertyValue_InContainer(Target, static_cast<int64>(Value));
        return true;
    }
    return false;
}

bool GetBoolLikeProperty(const UObject* Target, const TCHAR* PropertyName, bool& OutValue)
{
    if (Target == nullptr)
    {
        return false;
    }

    const FProperty* Property = FindPropertyByStableName(Target->GetClass(), PropertyName);
    if (Property == nullptr)
    {
        return false;
    }

    if (const FBoolProperty* BoolProperty = CastField<FBoolProperty>(Property))
    {
        OutValue = BoolProperty->GetPropertyValue_InContainer(Target);
        return true;
    }
    return false;
}

bool GetStringLikeProperty(const UObject* Target, const TCHAR* PropertyName, FString& OutValue)
{
    OutValue = TEXT("");
    if (Target == nullptr)
    {
        return false;
    }

    const FProperty* Property = FindPropertyByStableName(Target->GetClass(), PropertyName);
    if (Property == nullptr)
    {
        return false;
    }

    if (const FStrProperty* StringProperty = CastField<FStrProperty>(Property))
    {
        OutValue = StringProperty->GetPropertyValue_InContainer(Target);
        return true;
    }
    return false;
}

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

UDroneSensorsRuntimeComponent* ResolveSensorComponent(APawn* Pawn)
{
    if (Pawn == nullptr)
    {
        return nullptr;
    }

    TInlineComponentArray<UActorComponent*> Components(Pawn);
    for (UActorComponent* Component : Components)
    {
        if (UDroneSensorsRuntimeComponent* AsRuntime = Cast<UDroneSensorsRuntimeComponent>(Component))
        {
            return AsRuntime;
        }
    }

    for (UActorComponent* Component : Components)
    {
        if (Component == nullptr)
        {
            continue;
        }

        const FString Name = Component->GetName();
        const FString ClassName = Component->GetClass()->GetName();
        if (Name.Contains(TEXT("DroneSensors"), ESearchCase::IgnoreCase)
            || Name.Contains(TEXT("SensorsModule"), ESearchCase::IgnoreCase)
            || ClassName.Contains(TEXT("DroneSensors"), ESearchCase::IgnoreCase))
        {
            return Cast<UDroneSensorsRuntimeComponent>(Component);
        }
    }

    return nullptr;
}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FDroneSensorsRuntimeStep9BTest,
    "UE.Drone.ImageCapture.Step9B.DroneSensorsRuntimeCallable",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FDroneSensorsRuntimeStep9BTest::RunTest(const FString& Parameters)
{
#if !WITH_EDITOR
    AddError(TEXT("Step9B test requires editor context."));
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
    APawn* Pawn = World->SpawnActor<APawn>(
        PawnClass,
        FVector(0.0, 0.0, 200.0),
        FRotator::ZeroRotator,
        SpawnParameters
    );
    if (!TestNotNull(TEXT("Spawned BP_DronePawn"), Pawn))
    {
        return false;
    }

    const auto CleanupPawn = [&]()
    {
        if (IsValid(Pawn))
        {
            Pawn->Destroy();
        }
    };

    UDroneSensorsRuntimeComponent* Sensors = ResolveSensorComponent(Pawn);
    if (!TestNotNull(TEXT("Resolved UDroneSensorsRuntimeComponent"), Sensors))
    {
        CleanupPawn();
        return false;
    }

    // Positive configuration path.
    TestTrue(TEXT("Set active_viewpoint"), SetNameLikeProperty(Sensors, TEXT("active_viewpoint"), FName(TEXT("front"))));
    TestTrue(TEXT("Set front_fov_deg"), SetRealLikeProperty(Sensors, TEXT("front_fov_deg"), 77.5));
    TestTrue(TEXT("Set capture_width"), SetIntLikeProperty(Sensors, TEXT("capture_width"), 320));
    TestTrue(TEXT("Set capture_height"), SetIntLikeProperty(Sensors, TEXT("capture_height"), 200));
    TestTrue(TEXT("Set front_offset_x_cm"), SetRealLikeProperty(Sensors, TEXT("front_offset_x_cm"), 15.0));
    TestTrue(TEXT("Set front_offset_y_cm"), SetRealLikeProperty(Sensors, TEXT("front_offset_y_cm"), 0.0));
    TestTrue(TEXT("Set front_offset_z_cm"), SetRealLikeProperty(Sensors, TEXT("front_offset_z_cm"), 8.0));
    TestTrue(TEXT("Set front_rotation_pitch_deg"), SetRealLikeProperty(Sensors, TEXT("front_rotation_pitch_deg"), 0.0));
    TestTrue(TEXT("Set front_rotation_roll_deg"), SetRealLikeProperty(Sensors, TEXT("front_rotation_roll_deg"), 0.0));
    TestTrue(TEXT("Set front_rotation_yaw_deg"), SetRealLikeProperty(Sensors, TEXT("front_rotation_yaw_deg"), 0.0));

    Sensors->ApplySensorRigConfig();

    bool bRuntimeApplied = false;
    TestTrue(TEXT("Read runtime_config_applied"), GetBoolLikeProperty(Sensors, TEXT("runtime_config_applied"), bRuntimeApplied));
    TestTrue(TEXT("runtime_config_applied should be true after valid apply"), bRuntimeApplied);

    FString ApplyError;
    TestTrue(TEXT("Read config_apply_last_error"), GetStringLikeProperty(Sensors, TEXT("config_apply_last_error"), ApplyError));
    TestTrue(TEXT("config_apply_last_error should be empty after valid apply"), ApplyError.IsEmpty());

    TArray<uint8> PngBytes;
    int32 CaptureWidth = 0;
    int32 CaptureHeight = 0;
    FString CaptureError;
    const bool bCaptureOk = Sensors->CaptureActiveViewpointPngBytes(PngBytes, CaptureWidth, CaptureHeight, CaptureError);
    TestTrue(TEXT("CaptureActiveViewpointPngBytes should succeed"), bCaptureOk);
    TestTrue(TEXT("Capture error should be empty"), CaptureError.IsEmpty());
    TestTrue(TEXT("PNG bytes should be non-empty"), PngBytes.Num() > 0);
    TestEqual(TEXT("Capture width"), CaptureWidth, 320);
    TestEqual(TEXT("Capture height"), CaptureHeight, 200);

    int32 PngDecodedWidth = 0;
    int32 PngDecodedHeight = 0;
    TestTrue(TEXT("Decode PNG dimensions"), DecodePngDimensions(PngBytes, PngDecodedWidth, PngDecodedHeight));
    TestEqual(TEXT("PNG decoded width"), PngDecodedWidth, CaptureWidth);
    TestEqual(TEXT("PNG decoded height"), PngDecodedHeight, CaptureHeight);

    FDroneViewpointSnapshotNative Snapshot;
    FString SnapshotError;
    const bool bSnapshotOk = Sensors->GetViewpointSnapshot(
        TEXT("step9b_run"),
        TEXT("step9b_capture"),
        Snapshot,
        SnapshotError
    );
    TestTrue(TEXT("GetViewpointSnapshot should succeed"), bSnapshotOk);
    TestTrue(TEXT("Snapshot error should be empty"), SnapshotError.IsEmpty());
    TestEqual(TEXT("Snapshot viewpoint_name"), Snapshot.viewpoint_name.ToString(), FString(TEXT("front")));
    TestEqual(TEXT("Snapshot width"), Snapshot.width, CaptureWidth);
    TestEqual(TEXT("Snapshot height"), Snapshot.height, CaptureHeight);
    TestTrue(TEXT("Snapshot fov should match configured fov"), FMath::Abs(Snapshot.fov_deg - 77.5) < 0.001);

    // Negative path: invalid capture width must fail apply and capture.
    TestTrue(TEXT("Set invalid capture_width"), SetIntLikeProperty(Sensors, TEXT("capture_width"), 0));
    Sensors->ApplySensorRigConfig();

    bool bRuntimeAppliedAfterInvalid = true;
    TestTrue(
        TEXT("Read runtime_config_applied after invalid apply"),
        GetBoolLikeProperty(Sensors, TEXT("runtime_config_applied"), bRuntimeAppliedAfterInvalid)
    );
    TestFalse(TEXT("runtime_config_applied should be false after invalid apply"), bRuntimeAppliedAfterInvalid);

    FString InvalidApplyError;
    TestTrue(TEXT("Read config_apply_last_error after invalid apply"), GetStringLikeProperty(Sensors, TEXT("config_apply_last_error"), InvalidApplyError));
    TestTrue(TEXT("config_apply_last_error should be non-empty after invalid apply"), !InvalidApplyError.IsEmpty());

    TArray<uint8> NegativePngBytes;
    int32 NegativeWidth = 0;
    int32 NegativeHeight = 0;
    FString NegativeCaptureError;
    const bool bNegativeCaptureOk = Sensors->CaptureActiveViewpointPngBytes(
        NegativePngBytes,
        NegativeWidth,
        NegativeHeight,
        NegativeCaptureError
    );
    TestFalse(TEXT("CaptureActiveViewpointPngBytes should fail after invalid state"), bNegativeCaptureOk);
    TestTrue(TEXT("Negative capture error should be non-empty"), !NegativeCaptureError.IsEmpty());

    CleanupPawn();
    return !HasAnyErrors();
#endif
}

#endif
