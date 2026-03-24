// Status: in development.
// Runtime ingress/orchestration implementation. Transport ownership remains
// in UWSClientComponent while gameplay/runtime dispatch ownership lives here.

#include "SetDataConfigRuntimeActor.h"

#include "SampleManagerRuntimeActor.h"

#include "Components/ActorComponent.h"
#include "Dom/JsonObject.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Pawn.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "UObject/UnrealType.h"
#include "WSClientComponent.h"

namespace
{
static double ReadNumber(const TSharedPtr<FJsonObject>& PayloadObject, const TCHAR* FieldName, double DefaultValue = 0.0)
{
    if (!PayloadObject.IsValid())
    {
        return DefaultValue;
    }
    double Value = DefaultValue;
    if (PayloadObject->TryGetNumberField(FieldName, Value))
    {
        return Value;
    }
    return DefaultValue;
}

static bool TryReadNumber(const TSharedPtr<FJsonObject>& PayloadObject, const TCHAR* FieldName, double& OutValue)
{
    if (!PayloadObject.IsValid())
    {
        return false;
    }
    return PayloadObject->TryGetNumberField(FieldName, OutValue);
}

static FString CanonicalizeToken(FString Value)
{
    Value.ToLowerInline();
    Value.ReplaceInline(TEXT("_"), TEXT(""));
    return Value;
}

static FProperty* FindPropertyByStableName(UClass* Class, const TCHAR* PropertyName)
{
    if (Class == nullptr || PropertyName == nullptr || !*PropertyName)
    {
        return nullptr;
    }

    if (FProperty* Exact = Class->FindPropertyByName(FName(PropertyName)))
    {
        return Exact;
    }

    const FString Wanted(PropertyName);
    const FString WantedPrefix = Wanted + TEXT("_");
    const FString WantedCanonical = CanonicalizeToken(Wanted);
    for (TFieldIterator<FProperty> It(Class, EFieldIterationFlags::IncludeSuper); It; ++It)
    {
        FProperty* Candidate = *It;
        if (Candidate == nullptr)
        {
            continue;
        }
        const FString CandidateName = Candidate->GetName();
        const FString CandidateCanonical = CanonicalizeToken(CandidateName);
        if (CandidateName.Equals(Wanted, ESearchCase::IgnoreCase)
            || CandidateName.StartsWith(WantedPrefix, ESearchCase::IgnoreCase)
            || CandidateCanonical.Equals(WantedCanonical, ESearchCase::CaseSensitive)
            || CandidateCanonical.StartsWith(WantedCanonical, ESearchCase::CaseSensitive))
        {
            return Candidate;
        }
    }
    return nullptr;
}
}

ASetDataConfigRuntimeActor::ASetDataConfigRuntimeActor()
{
    PrimaryActorTick.bCanEverTick = false;
    ServerUrl = TEXT("ws://127.0.0.1:8765");
    WSClient = CreateDefaultSubobject<UWSClientComponent>(TEXT("WSClient"));
}

void ASetDataConfigRuntimeActor::BeginPlay()
{
    Super::BeginPlay();

    if (!WSClient)
    {
        LastError = TEXT("WSClient component missing.");
        UE_LOG(LogTemp, Error, TEXT("[SetDataConfigRuntimeActor] %s"), *LastError);
        return;
    }

    WSClient->OnConnected.AddDynamic(this, &ASetDataConfigRuntimeActor::HandleSocketConnected);
    WSClient->OnClosed.AddDynamic(this, &ASetDataConfigRuntimeActor::HandleSocketClosed);
    WSClient->OnError.AddDynamic(this, &ASetDataConfigRuntimeActor::HandleSocketError);
    WSClient->OnMessage.AddDynamic(this, &ASetDataConfigRuntimeActor::HandleSocketMessage);

    if (bAutoConnectOnBeginPlay)
    {
        Connect();
    }
}

void ASetDataConfigRuntimeActor::Connect()
{
    if (!WSClient)
    {
        return;
    }
    WSClient->SetServerUrl(ServerUrl);
    WSClient->Connect();
}

void ASetDataConfigRuntimeActor::Disconnect()
{
    if (WSClient)
    {
        WSClient->Disconnect();
    }
}

bool ASetDataConfigRuntimeActor::OnSetConfigReceived_Implementation(
    const FString& RunId,
    const FString& ConfigId,
    const FString& ConfigHash,
    const FString& PayloadJson
)
{
    ActiveRunId = RunId;

    TSharedPtr<FJsonObject> PayloadObject;
    FString ParseError;
    if (!ParseRootObject(PayloadJson, PayloadObject, ParseError))
    {
        EmitConfigError(FString::Printf(TEXT("SET_CONFIG payload parse failed: %s"), *ParseError));
        InvokeBlueprintHookIfExists(TEXT("OnConfigRejected"));
        return false;
    }

    FSensorRigRuntimeConfig ParsedConfig;
    FString ValidationError;
    if (!ExtractSensorRigConfig(PayloadObject, ParsedConfig, ValidationError))
    {
        EmitConfigError(ValidationError);
        InvokeBlueprintHookIfExists(TEXT("OnConfigRejected"));
        return false;
    }

    UWorld* World = GetWorld();
    if (World == nullptr)
    {
        EmitConfigError(TEXT("World is null during config apply."));
        InvokeBlueprintHookIfExists(TEXT("OnConfigRejected"));
        return false;
    }

    ActiveSensorRigConfig = ParsedConfig;

    int32 SensorTargetsFound = 0;
    int32 SensorTargetsApplied = 0;
    for (TActorIterator<APawn> It(World); It; ++It)
    {
        APawn* CandidatePawn = *It;
        if (CandidatePawn == nullptr)
        {
            continue;
        }
        UActorComponent* SensorTarget = ResolveSensorComponent(CandidatePawn);
        if (SensorTarget == nullptr)
        {
            continue;
        }

        ++SensorTargetsFound;
        FString ApplyError;
        if (!ApplySensorRigConfigToTarget(SensorTarget, ApplyError))
        {
            EmitConfigError(
                FString::Printf(
                    TEXT("SET_CONFIG apply failed target=%s owner=%s reason=%s"),
                    *SensorTarget->GetName(),
                    *CandidatePawn->GetName(),
                    *ApplyError
                )
            );
            InvokeBlueprintHookIfExists(TEXT("OnConfigRejected"));
            return false;
        }
        ++SensorTargetsApplied;
    }

    if (SensorTargetsFound == 0)
    {
        bHasActiveSensorRigConfig = false;
        EmitConfigError(
            FString::Printf(
                TEXT("SET_CONFIG rejected: no live sensor owners found for run_id=%s config_id=%s"),
                *RunId,
                *ConfigId
            )
        );
        InvokeBlueprintHookIfExists(TEXT("OnConfigRejected"));
        return false;
    }

    ActiveConfigId = ConfigId;
    ActiveConfigHash = ConfigHash;
    bHasActiveSensorRigConfig = true;

    UE_LOG(
        LogTemp,
        Display,
        TEXT("[SetDataConfigRuntimeActor] set_config applied to %d/%d live sensor owners"),
        SensorTargetsApplied,
        SensorTargetsFound
    );

    EmitConfigAck();
    EmitConfigReady();
    InvokeBlueprintHookIfExists(TEXT("OnConfigAccepted"));
    return bConfigReady;
}

bool ASetDataConfigRuntimeActor::GetActiveConfigReference(FString& OutConfigId, FString& OutConfigHash) const
{
    OutConfigId = ActiveConfigId;
    OutConfigHash = ActiveConfigHash;
    return bConfigReady && !ActiveConfigId.IsEmpty() && !ActiveConfigHash.IsEmpty();
}

void ASetDataConfigRuntimeActor::ClearActiveConfigReference()
{
    ActiveConfigId = TEXT("");
    ActiveConfigHash = TEXT("");
    bHasActiveSensorRigConfig = false;
}

void ASetDataConfigRuntimeActor::InvokeBlueprintHookIfExists(const TCHAR* FunctionName)
{
    if (FunctionName == nullptr || !*FunctionName)
    {
        return;
    }
    if (UFunction* Hook = FindFunction(FName(FunctionName)))
    {
        ProcessEvent(Hook, nullptr);
    }
}

bool ASetDataConfigRuntimeActor::ExtractSensorRigConfig(
    const TSharedPtr<FJsonObject>& PayloadObject,
    FSensorRigRuntimeConfig& OutConfig,
    FString& OutError
) const
{
    if (!PayloadObject.IsValid())
    {
        OutError = TEXT("SET_CONFIG payload object is invalid.");
        return false;
    }

    const TSharedPtr<FJsonObject>* SensorRigObject = nullptr;
    if (!PayloadObject->TryGetObjectField(TEXT("sensor_rig"), SensorRigObject) || SensorRigObject == nullptr || !SensorRigObject->IsValid())
    {
        OutError = TEXT("SET_CONFIG payload missing sensor_rig object.");
        return false;
    }

    FString ActiveViewpoint;
    if (!TryGetStringField(*SensorRigObject, TEXT("active_viewpoint"), ActiveViewpoint))
    {
        OutError = TEXT("SET_CONFIG payload missing sensor_rig.active_viewpoint.");
        return false;
    }
    if (!ActiveViewpoint.Equals(TEXT("front"), ESearchCase::IgnoreCase))
    {
        OutError = FString::Printf(TEXT("Unsupported sensor_rig.active_viewpoint: %s"), *ActiveViewpoint);
        return false;
    }

    double FrontFovDeg = 0.0;
    double CaptureWidthValue = 0.0;
    double CaptureHeightValue = 0.0;
    if (!TryReadNumber(*SensorRigObject, TEXT("front_fov_deg"), FrontFovDeg))
    {
        OutError = TEXT("SET_CONFIG payload missing sensor_rig.front_fov_deg.");
        return false;
    }
    if (!TryReadNumber(*SensorRigObject, TEXT("capture_width"), CaptureWidthValue))
    {
        OutError = TEXT("SET_CONFIG payload missing sensor_rig.capture_width.");
        return false;
    }
    if (!TryReadNumber(*SensorRigObject, TEXT("capture_height"), CaptureHeightValue))
    {
        OutError = TEXT("SET_CONFIG payload missing sensor_rig.capture_height.");
        return false;
    }

    const int32 CaptureWidth = FMath::RoundToInt(static_cast<float>(CaptureWidthValue));
    const int32 CaptureHeight = FMath::RoundToInt(static_cast<float>(CaptureHeightValue));
    if (FrontFovDeg <= 0.0)
    {
        OutError = TEXT("sensor_rig.front_fov_deg must be > 0.");
        return false;
    }
    if (CaptureWidth <= 0 || CaptureHeight <= 0)
    {
        OutError = TEXT("sensor_rig.capture_width/capture_height must be positive integers.");
        return false;
    }

    const TSharedPtr<FJsonObject>* FrontOffsetObject = nullptr;
    if (!(*SensorRigObject)->TryGetObjectField(TEXT("front_offset_cm"), FrontOffsetObject)
        || FrontOffsetObject == nullptr
        || !FrontOffsetObject->IsValid())
    {
        OutError = TEXT("SET_CONFIG payload missing sensor_rig.front_offset_cm object.");
        return false;
    }

    const TSharedPtr<FJsonObject>* FrontRotationObject = nullptr;
    if (!(*SensorRigObject)->TryGetObjectField(TEXT("front_rotation_deg"), FrontRotationObject)
        || FrontRotationObject == nullptr
        || !FrontRotationObject->IsValid())
    {
        OutError = TEXT("SET_CONFIG payload missing sensor_rig.front_rotation_deg object.");
        return false;
    }

    double OffsetX = 0.0;
    double OffsetY = 0.0;
    double OffsetZ = 0.0;
    double RotationPitch = 0.0;
    double RotationRoll = 0.0;
    double RotationYaw = 0.0;
    if (!TryReadNumber(*FrontOffsetObject, TEXT("x"), OffsetX)
        || !TryReadNumber(*FrontOffsetObject, TEXT("y"), OffsetY)
        || !TryReadNumber(*FrontOffsetObject, TEXT("z"), OffsetZ))
    {
        OutError = TEXT("SET_CONFIG payload missing one or more sensor_rig.front_offset_cm fields.");
        return false;
    }
    if (!TryReadNumber(*FrontRotationObject, TEXT("pitch"), RotationPitch)
        || !TryReadNumber(*FrontRotationObject, TEXT("roll"), RotationRoll)
        || !TryReadNumber(*FrontRotationObject, TEXT("yaw"), RotationYaw))
    {
        OutError = TEXT("SET_CONFIG payload missing one or more sensor_rig.front_rotation_deg fields.");
        return false;
    }

    OutConfig.ActiveViewpoint = FName(*ActiveViewpoint);
    OutConfig.FrontFovDeg = FrontFovDeg;
    OutConfig.CaptureWidth = CaptureWidth;
    OutConfig.CaptureHeight = CaptureHeight;
    OutConfig.FrontOffsetCm = FVector(OffsetX, OffsetY, OffsetZ);
    OutConfig.FrontRotationDeg = FRotator(RotationPitch, RotationYaw, RotationRoll);
    OutError = TEXT("");
    return true;
}

bool ASetDataConfigRuntimeActor::ReadPrimarySensorComponentName(APawn* Pawn, FString& OutComponentName) const
{
    OutComponentName = TEXT("");
    if (Pawn == nullptr)
    {
        return false;
    }

    FProperty* Property = FindPropertyByStableName(Pawn->GetClass(), TEXT("primary_sensor_component_name"));
    if (Property == nullptr)
    {
        return false;
    }
    if (const FStrProperty* StrProperty = CastField<FStrProperty>(Property))
    {
        OutComponentName = StrProperty->GetPropertyValue_InContainer(Pawn);
        return !OutComponentName.IsEmpty();
    }
    if (const FNameProperty* NameProperty = CastField<FNameProperty>(Property))
    {
        OutComponentName = NameProperty->GetPropertyValue_InContainer(Pawn).ToString();
        return !OutComponentName.IsEmpty();
    }
    return false;
}

UActorComponent* ASetDataConfigRuntimeActor::ResolveSensorComponent(APawn* Pawn) const
{
    if (Pawn == nullptr)
    {
        return nullptr;
    }

    TInlineComponentArray<UActorComponent*> Components(Pawn);
    FString PreferredComponentName;
    const bool bHasPreferredName = ReadPrimarySensorComponentName(Pawn, PreferredComponentName);

    if (bHasPreferredName)
    {
        for (UActorComponent* Component : Components)
        {
            if (Component == nullptr)
            {
                continue;
            }
            const FString ComponentName = Component->GetName();
            if (ComponentName.Equals(PreferredComponentName, ESearchCase::IgnoreCase))
            {
                return Component;
            }
        }
    }

    for (UActorComponent* Component : Components)
    {
        if (Component == nullptr)
        {
            continue;
        }
        const FString ComponentClassName = Component->GetClass()->GetName();
        const FString ComponentName = Component->GetName();
        if (ComponentClassName.Contains(TEXT("DroneSensors"), ESearchCase::IgnoreCase)
            || ComponentName.Contains(TEXT("DroneSensors"), ESearchCase::IgnoreCase)
            || ComponentName.Contains(TEXT("SensorsModule"), ESearchCase::IgnoreCase))
        {
            return Component;
        }
    }

    for (UActorComponent* Component : Components)
    {
        if (Component == nullptr)
        {
            continue;
        }
        if (Component->GetClass()->FindPropertyByName(FName(TEXT("front_fov_deg"))) != nullptr
            && Component->GetClass()->FindPropertyByName(FName(TEXT("capture_width"))) != nullptr
            && Component->GetClass()->FindPropertyByName(FName(TEXT("capture_height"))) != nullptr)
        {
            return Component;
        }
    }

    return nullptr;
}

bool ASetDataConfigRuntimeActor::ApplySensorRigConfigToTarget(UObject* Target, FString& OutError) const
{
    OutError = TEXT("");
    if (Target == nullptr)
    {
        OutError = TEXT("Target is null.");
        return false;
    }

    const bool bRequiredApplied = SetNameLikeProperty(Target, TEXT("active_viewpoint"), ActiveSensorRigConfig.ActiveViewpoint)
        && SetRealLikeProperty(Target, TEXT("front_fov_deg"), ActiveSensorRigConfig.FrontFovDeg)
        && SetIntLikeProperty(Target, TEXT("capture_width"), ActiveSensorRigConfig.CaptureWidth)
        && SetIntLikeProperty(Target, TEXT("capture_height"), ActiveSensorRigConfig.CaptureHeight)
        && SetRealLikeProperty(Target, TEXT("front_offset_x_cm"), ActiveSensorRigConfig.FrontOffsetCm.X)
        && SetRealLikeProperty(Target, TEXT("front_offset_y_cm"), ActiveSensorRigConfig.FrontOffsetCm.Y)
        && SetRealLikeProperty(Target, TEXT("front_offset_z_cm"), ActiveSensorRigConfig.FrontOffsetCm.Z)
        && SetRealLikeProperty(Target, TEXT("front_rotation_pitch_deg"), ActiveSensorRigConfig.FrontRotationDeg.Pitch)
        && SetRealLikeProperty(Target, TEXT("front_rotation_roll_deg"), ActiveSensorRigConfig.FrontRotationDeg.Roll)
        && SetRealLikeProperty(Target, TEXT("front_rotation_yaw_deg"), ActiveSensorRigConfig.FrontRotationDeg.Yaw);

    if (!bRequiredApplied)
    {
        OutError = FString::Printf(
            TEXT("Target missing one or more required sensor-rig properties class=%s name=%s"),
            *Target->GetClass()->GetName(),
            *Target->GetName()
        );
        return false;
    }

    if (!SetBoolLikeProperty(Target, TEXT("runtime_config_applied"), false))
    {
        OutError = FString::Printf(
            TEXT("Target missing runtime_config_applied property class=%s name=%s"),
            *Target->GetClass()->GetName(),
            *Target->GetName()
        );
        return false;
    }

    UFunction* ApplyFunction = Target->FindFunction(FName(TEXT("ApplySensorRigConfig")));
    if (ApplyFunction == nullptr)
    {
        OutError = FString::Printf(
            TEXT("Target missing ApplySensorRigConfig function class=%s name=%s"),
            *Target->GetClass()->GetName(),
            *Target->GetName()
        );
        return false;
    }
    Target->ProcessEvent(ApplyFunction, nullptr);

    bool bRuntimeConfigApplied = false;
    if (!GetBoolLikeProperty(Target, TEXT("runtime_config_applied"), bRuntimeConfigApplied))
    {
        OutError = FString::Printf(
            TEXT("Target missing runtime_config_applied property after apply class=%s name=%s"),
            *Target->GetClass()->GetName(),
            *Target->GetName()
        );
        return false;
    }
    if (!bRuntimeConfigApplied)
    {
        FString ApplyErrorMessage;
        if (GetStringLikeProperty(Target, TEXT("config_apply_last_error"), ApplyErrorMessage) && !ApplyErrorMessage.IsEmpty())
        {
            OutError = FString::Printf(TEXT("ApplySensorRigConfig rejected runtime config: %s"), *ApplyErrorMessage);
        }
        else
        {
            OutError = TEXT("ApplySensorRigConfig rejected runtime config.");
        }
        return false;
    }

    return true;
}

bool ASetDataConfigRuntimeActor::ApplySensorRigConfigToPawn(APawn* Pawn, FString& OutError) const
{
    if (Pawn == nullptr)
    {
        OutError = TEXT("Pawn is null.");
        return false;
    }

    if (UActorComponent* SensorTarget = ResolveSensorComponent(Pawn))
    {
        return ApplySensorRigConfigToTarget(SensorTarget, OutError);
    }

    return ApplySensorRigConfigToTarget(Pawn, OutError);
}

void ASetDataConfigRuntimeActor::ApplySensorRigConfigToPawnIfAvailable(const FString& RunId, APawn* Pawn)
{
    if (!bHasActiveSensorRigConfig || Pawn == nullptr)
    {
        return;
    }

    FString ApplyError;
    if (!ApplySensorRigConfigToPawn(Pawn, ApplyError))
    {
        UE_LOG(
            LogTemp,
            Warning,
            TEXT("[SetDataConfigRuntimeActor] failed to apply cached sensor config to spawned pawn %s: %s"),
            *Pawn->GetName(),
            *ApplyError
        );
        SendError(RunId, FString::Printf(TEXT("cached sensor config apply failed for spawned pawn %s: %s"), *Pawn->GetName(), *ApplyError));
    }
}

bool ASetDataConfigRuntimeActor::SetNameLikeProperty(UObject* Target, const TCHAR* PropertyName, const FName& Value) const
{
    if (Target == nullptr || PropertyName == nullptr || !*PropertyName)
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

bool ASetDataConfigRuntimeActor::SetRealLikeProperty(UObject* Target, const TCHAR* PropertyName, double Value) const
{
    if (Target == nullptr || PropertyName == nullptr || !*PropertyName)
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

bool ASetDataConfigRuntimeActor::SetIntLikeProperty(UObject* Target, const TCHAR* PropertyName, int32 Value) const
{
    if (Target == nullptr || PropertyName == nullptr || !*PropertyName)
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

bool ASetDataConfigRuntimeActor::SetBoolLikeProperty(UObject* Target, const TCHAR* PropertyName, bool Value) const
{
    if (Target == nullptr || PropertyName == nullptr || !*PropertyName)
    {
        return false;
    }
    FProperty* Property = FindPropertyByStableName(Target->GetClass(), PropertyName);
    if (Property == nullptr)
    {
        return false;
    }
    if (FBoolProperty* BoolProperty = CastField<FBoolProperty>(Property))
    {
        BoolProperty->SetPropertyValue_InContainer(Target, Value);
        return true;
    }
    return false;
}

bool ASetDataConfigRuntimeActor::GetBoolLikeProperty(const UObject* Target, const TCHAR* PropertyName, bool& OutValue) const
{
    if (Target == nullptr || PropertyName == nullptr || !*PropertyName)
    {
        return false;
    }
    const FProperty* Property = FindPropertyByStableName(const_cast<UClass*>(Target->GetClass()), PropertyName);
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

bool ASetDataConfigRuntimeActor::GetStringLikeProperty(const UObject* Target, const TCHAR* PropertyName, FString& OutValue) const
{
    if (Target == nullptr || PropertyName == nullptr || !*PropertyName)
    {
        return false;
    }
    const FProperty* Property = FindPropertyByStableName(const_cast<UClass*>(Target->GetClass()), PropertyName);
    if (Property == nullptr)
    {
        return false;
    }
    if (const FStrProperty* StringProperty = CastField<FStrProperty>(Property))
    {
        OutValue = StringProperty->GetPropertyValue_InContainer(Target);
        return true;
    }
    if (const FNameProperty* NameProperty = CastField<FNameProperty>(Property))
    {
        OutValue = NameProperty->GetPropertyValue_InContainer(Target).ToString();
        return true;
    }
    return false;
}

void ASetDataConfigRuntimeActor::EmitConfigAck()
{
    if (ActiveRunId.IsEmpty())
    {
        LastError = TEXT("Cannot emit ACK: active run id is empty.");
        UE_LOG(LogTemp, Error, TEXT("[SetDataConfigRuntimeActor] %s"), *LastError);
        return;
    }

    TSharedPtr<FJsonObject> AckPayload = MakeShared<FJsonObject>();
    AckPayload->SetBoolField(TEXT("received"), true);
    AckPayload->SetStringField(TEXT("config_id"), ActiveConfigId);
    AckPayload->SetStringField(TEXT("config_hash"), ActiveConfigHash);
    AckPayload->SetStringField(TEXT("source"), TEXT("BP_SetDataConfig"));
    SendEnvelope(TEXT("ACK"), ActiveRunId, AckPayload);
    bConfigResponseSentForCurrentSetConfig = true;
}

void ASetDataConfigRuntimeActor::EmitConfigReady()
{
    if (ActiveRunId.IsEmpty() || ActiveConfigId.IsEmpty() || ActiveConfigHash.IsEmpty())
    {
        LastError = TEXT("Cannot emit CONFIG_READY: active config identity is incomplete.");
        UE_LOG(LogTemp, Error, TEXT("[SetDataConfigRuntimeActor] %s"), *LastError);
        return;
    }

    TSharedPtr<FJsonObject> ReadyPayload = MakeShared<FJsonObject>();
    ReadyPayload->SetStringField(TEXT("config_id"), ActiveConfigId);
    ReadyPayload->SetStringField(TEXT("config_hash"), ActiveConfigHash);
    ReadyPayload->SetStringField(TEXT("source"), TEXT("BP_SetDataConfig"));
    SendEnvelope(TEXT("CONFIG_READY"), ActiveRunId, ReadyPayload);
    bConfigResponseSentForCurrentSetConfig = true;

    bConfigReady = true;
    LastError = TEXT("");
    UE_LOG(
        LogTemp,
        Display,
        TEXT("[SetDataConfigRuntimeActor] config ready run_id=%s config_id=%s config_hash=%s"),
        *ActiveRunId,
        *ActiveConfigId,
        *ActiveConfigHash
    );
}

void ASetDataConfigRuntimeActor::EmitConfigError(const FString& ErrorMessage)
{
    if (ActiveRunId.IsEmpty())
    {
        LastError = TEXT("Cannot emit config error: active run id is empty.");
        UE_LOG(LogTemp, Error, TEXT("[SetDataConfigRuntimeActor] %s"), *LastError);
        return;
    }

    const FString ResolvedMessage = ErrorMessage.IsEmpty() ? TEXT("Config apply rejected.") : ErrorMessage;
    LastError = ResolvedMessage;
    bConfigReady = false;
    ClearActiveConfigReference();
    SendError(ActiveRunId, ResolvedMessage);
    bConfigResponseSentForCurrentSetConfig = true;
}

void ASetDataConfigRuntimeActor::HandleSocketConnected()
{
    UE_LOG(LogTemp, Display, TEXT("[SetDataConfigRuntimeActor] connected url=%s"), *ServerUrl);
}

void ASetDataConfigRuntimeActor::HandleSocketClosed(int32 StatusCode, const FString& Reason)
{
    UE_LOG(LogTemp, Warning, TEXT("[SetDataConfigRuntimeActor] closed status=%d reason=%s"), StatusCode, *Reason);
}

void ASetDataConfigRuntimeActor::HandleSocketError(const FString& ErrorText)
{
    LastError = ErrorText;
    UE_LOG(LogTemp, Error, TEXT("[SetDataConfigRuntimeActor] socket error: %s"), *ErrorText);
}

void ASetDataConfigRuntimeActor::HandleSocketMessage(const FString& MessageText)
{
    TSharedPtr<FJsonObject> Root;
    FString ParseError;
    if (!ParseRootObject(MessageText, Root, ParseError))
    {
        LastError = ParseError;
        UE_LOG(LogTemp, Error, TEXT("[SetDataConfigRuntimeActor] parse error: %s"), *ParseError);
        return;
    }

    FString MessageType;
    FString RunId;
    if (!TryGetStringField(Root, TEXT("type"), MessageType) || !TryGetStringField(Root, TEXT("run_id"), RunId))
    {
        LastError = TEXT("Incoming message missing required fields: type/run_id.");
        UE_LOG(LogTemp, Error, TEXT("[SetDataConfigRuntimeActor] %s"), *LastError);
        return;
    }

    if (!MessageType.Equals(TEXT("SET_CONFIG"), ESearchCase::IgnoreCase))
    {
        const TSharedPtr<FJsonObject>* PayloadObject = nullptr;
        if (!Root->TryGetObjectField(TEXT("payload"), PayloadObject) || PayloadObject == nullptr || !PayloadObject->IsValid())
        {
            if (MessageType.Equals(TEXT("SPAWN_DRONES"), ESearchCase::IgnoreCase)
                || MessageType.Equals(TEXT("CMD"), ESearchCase::IgnoreCase)
                || MessageType.Equals(TEXT("CAPTURE_NOW"), ESearchCase::IgnoreCase))
            {
                SendError(RunId, TEXT("Action payload object is required."));
            }
            return;
        }
        HandleActionMessage(MessageType, RunId, Root, *PayloadObject);
        return;
    }

    if (!ExpectedRunId.IsEmpty() && !RunId.Equals(ExpectedRunId, ESearchCase::CaseSensitive))
    {
        const FString ErrorText = FString::Printf(
            TEXT("RunId mismatch expected=%s received=%s"),
            *ExpectedRunId,
            *RunId
        );
        LastError = ErrorText;
        UE_LOG(LogTemp, Error, TEXT("[SetDataConfigRuntimeActor] %s"), *ErrorText);
        SendError(RunId, ErrorText);
        if (bFailOnRunIdMismatch)
        {
            return;
        }
    }

    const TSharedPtr<FJsonObject>* PayloadObject = nullptr;
    if (!Root->TryGetObjectField(TEXT("payload"), PayloadObject) || PayloadObject == nullptr || !PayloadObject->IsValid())
    {
        LastError = TEXT("SET_CONFIG missing payload object.");
        UE_LOG(LogTemp, Error, TEXT("[SetDataConfigRuntimeActor] %s"), *LastError);
        SendError(RunId, LastError);
        return;
    }

    FString ConfigId;
    FString ConfigHash;
    if (!TryGetStringField(*PayloadObject, TEXT("config_id"), ConfigId) || !TryGetStringField(*PayloadObject, TEXT("config_hash"), ConfigHash))
    {
        LastError = TEXT("SET_CONFIG payload missing config_id/config_hash.");
        UE_LOG(LogTemp, Error, TEXT("[SetDataConfigRuntimeActor] %s"), *LastError);
        SendError(RunId, LastError);
        return;
    }

    bConfigReceived = true;
    bConfigReady = false;
    bConfigResponseSentForCurrentSetConfig = false;
    ActiveRunId = RunId;
    ClearActiveConfigReference();

    FString PayloadJson = TEXT("{}");
    {
        const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&PayloadJson);
        FJsonSerializer::Serialize(PayloadObject->ToSharedRef(), Writer);
    }

    LastError = TEXT("");
    const bool bAcceptedForApply = OnSetConfigReceived(RunId, ConfigId, ConfigHash, PayloadJson);
    if (!bAcceptedForApply)
    {
        if (LastError.IsEmpty())
        {
            LastError = TEXT("SET_CONFIG was not accepted by BP_SetDataConfig.");
        }
        UE_LOG(LogTemp, Error, TEXT("[SetDataConfigRuntimeActor] %s"), *LastError);
        if (!bConfigResponseSentForCurrentSetConfig)
        {
            SendError(RunId, LastError);
        }
        return;
    }

    if (!bConfigReady)
    {
        UE_LOG(
            LogTemp,
            Display,
            TEXT("[SetDataConfigRuntimeActor] set_config handed off run_id=%s config_id=%s config_hash=%s awaiting explicit config-ready response"),
            *ActiveRunId,
            *ActiveConfigId,
            *ActiveConfigHash
        );
    }
}

void ASetDataConfigRuntimeActor::SendEnvelope(
    const FString& Type,
    const FString& RunId,
    const TSharedPtr<FJsonObject>& PayloadObject,
    const FString& DroneId,
    const FString& CaptureId
)
{
    if (!WSClient || !WSClient->IsConnected())
    {
        UE_LOG(LogTemp, Warning, TEXT("[SetDataConfigRuntimeActor] cannot send %s: websocket not connected"), *Type);
        return;
    }

    FString PayloadJson = TEXT("{}");
    if (PayloadObject.IsValid())
    {
        const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&PayloadJson);
        FJsonSerializer::Serialize(PayloadObject.ToSharedRef(), Writer);
    }

    WSClient->SendEnvelopeJson(
        Type,
        RunId,
        PayloadJson,
        OutSeq++,
        DroneId,
        CaptureId
    );
}

void ASetDataConfigRuntimeActor::SendError(const FString& RunId, const FString& ErrorMessage)
{
    TSharedPtr<FJsonObject> ErrorPayload = MakeShared<FJsonObject>();
    ErrorPayload->SetStringField(TEXT("message"), ErrorMessage);
    ErrorPayload->SetStringField(TEXT("source"), TEXT("BP_SetDataConfig"));
    SendEnvelope(TEXT("ERROR"), RunId, ErrorPayload);
}

bool ASetDataConfigRuntimeActor::ParseRootObject(
    const FString& MessageText,
    TSharedPtr<FJsonObject>& OutRoot,
    FString& OutError
) const
{
    const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(MessageText);
    if (!FJsonSerializer::Deserialize(Reader, OutRoot) || !OutRoot.IsValid())
    {
        OutError = TEXT("Failed to parse message JSON.");
        return false;
    }
    OutError = TEXT("");
    return true;
}

bool ASetDataConfigRuntimeActor::TryGetStringField(const TSharedPtr<FJsonObject>& Root, const TCHAR* Key, FString& OutValue) const
{
    if (!Root.IsValid())
    {
        return false;
    }
    FString Parsed;
    if (!Root->TryGetStringField(Key, Parsed) || Parsed.IsEmpty())
    {
        return false;
    }
    OutValue = Parsed;
    return true;
}

ASampleManagerRuntimeActor* ASetDataConfigRuntimeActor::ResolveSampleManagerActor(FString& OutError) const
{
    OutError = TEXT("");
    UWorld* World = GetWorld();
    if (World == nullptr)
    {
        OutError = TEXT("CAPTURE_NOW dispatch failed: world is null.");
        return nullptr;
    }

    TArray<ASampleManagerRuntimeActor*> TypedCandidates;
    TArray<ASampleManagerRuntimeActor*> RoleCandidates;
#if WITH_EDITOR
    TArray<ASampleManagerRuntimeActor*> LabelCandidates;
#endif

    for (TActorIterator<ASampleManagerRuntimeActor> It(World); It; ++It)
    {
        ASampleManagerRuntimeActor* Candidate = *It;
        if (Candidate == nullptr)
        {
            continue;
        }

        TypedCandidates.Add(Candidate);

        if (Candidate->runtime_discovery_role.Equals(TEXT("sample_orchestrator"), ESearchCase::IgnoreCase))
        {
            RoleCandidates.Add(Candidate);
        }

#if WITH_EDITOR
        if (Candidate->GetActorLabel().Equals(TEXT("BP_SampleManager_Main"), ESearchCase::IgnoreCase))
        {
            LabelCandidates.Add(Candidate);
        }
#endif
    }

    if (RoleCandidates.Num() == 1)
    {
        return RoleCandidates[0];
    }
    if (RoleCandidates.Num() > 1)
    {
        OutError = FString::Printf(
            TEXT("CAPTURE_NOW dispatch failed: multiple sample managers matched runtime_discovery_role=sample_orchestrator count=%d."),
            RoleCandidates.Num()
        );
        return nullptr;
    }

#if WITH_EDITOR
    if (LabelCandidates.Num() == 1)
    {
        return LabelCandidates[0];
    }
    if (LabelCandidates.Num() > 1)
    {
        OutError = FString::Printf(
            TEXT("CAPTURE_NOW dispatch failed: multiple sample managers matched actor_label=BP_SampleManager_Main count=%d."),
            LabelCandidates.Num()
        );
        return nullptr;
    }
#endif

    if (TypedCandidates.Num() == 1)
    {
        return TypedCandidates[0];
    }
    if (TypedCandidates.Num() == 0)
    {
        OutError = TEXT("CAPTURE_NOW dispatch failed: no ASampleManagerRuntimeActor was found.");
        return nullptr;
    }

    OutError = FString::Printf(
        TEXT("CAPTURE_NOW dispatch failed: multiple sample manager actors were found count=%d."),
        TypedCandidates.Num()
    );
    return nullptr;
}

bool ASetDataConfigRuntimeActor::InvokeSampleManagerCaptureNow(
    ASampleManagerRuntimeActor* Target,
    const FString& RunId,
    const FString& DroneId,
    const FString& CaptureId,
    TSharedPtr<FJsonObject>& OutObservationPayload,
    FString& OutResolvedDroneId,
    FString& OutResolvedCaptureId,
    FString& OutError
) const
{
    OutObservationPayload.Reset();
    OutResolvedDroneId = TEXT("");
    OutResolvedCaptureId = TEXT("");
    OutError = TEXT("");

    if (Target == nullptr)
    {
        OutError = TEXT("CAPTURE_NOW dispatch failed: sample manager target is null.");
        return false;
    }

    FString ObservationJson;
    FString CaptureError;
    if (!Target->CaptureNow(RunId, DroneId, CaptureId, ObservationJson, CaptureError))
    {
        OutError = CaptureError.IsEmpty()
            ? FString::Printf(TEXT("CAPTURE_NOW dispatch failed: CaptureNow rejected on %s."), *Target->GetName())
            : CaptureError;
        return false;
    }

    if (ObservationJson.IsEmpty())
    {
        OutError = FString::Printf(
            TEXT("CAPTURE_NOW dispatch failed: CaptureNow on %s returned empty observation JSON."),
            *Target->GetName()
        );
        return false;
    }

    TSharedPtr<FJsonObject> ObservationPayload;
    FString ParseError;
    if (!ParseRootObject(ObservationJson, ObservationPayload, ParseError))
    {
        OutError = FString::Printf(
            TEXT("CAPTURE_NOW dispatch failed: observation parse failed from %s: %s"),
            *Target->GetName(),
            *ParseError
        );
        return false;
    }
    if (!ObservationPayload.IsValid())
    {
        OutError = FString::Printf(
            TEXT("CAPTURE_NOW dispatch failed: observation payload was invalid from %s."),
            *Target->GetName()
        );
        return false;
    }

    const TSharedPtr<FJsonObject>* ConfigRefObject = nullptr;
    if (!ObservationPayload->TryGetObjectField(TEXT("config_ref"), ConfigRefObject)
        || ConfigRefObject == nullptr
        || !ConfigRefObject->IsValid())
    {
        OutError = FString::Printf(
            TEXT("CAPTURE_NOW dispatch failed: sample manager observation missing config_ref object (%s)."),
            *Target->GetName()
        );
        return false;
    }

    const TSharedPtr<FJsonObject>* ViewpointObject = nullptr;
    if (!ObservationPayload->TryGetObjectField(TEXT("viewpoint"), ViewpointObject)
        || ViewpointObject == nullptr
        || !ViewpointObject->IsValid())
    {
        OutError = FString::Printf(
            TEXT("CAPTURE_NOW dispatch failed: sample manager observation missing viewpoint object (%s)."),
            *Target->GetName()
        );
        return false;
    }

    const TSharedPtr<FJsonObject>* ImageObject = nullptr;
    bool bHasCanonicalImageBytes = false;
    if (ObservationPayload->TryGetObjectField(TEXT("image"), ImageObject)
        && ImageObject != nullptr
        && ImageObject->IsValid())
    {
        FString ImageBytesB64;
        bHasCanonicalImageBytes = (*ImageObject)->TryGetStringField(TEXT("bytes_b64"), ImageBytesB64) && !ImageBytesB64.IsEmpty();
    }
    FString LegacyImageBytesB64;
    const bool bHasLegacyImageBytes = ObservationPayload->TryGetStringField(TEXT("image_bytes_b64"), LegacyImageBytesB64)
        && !LegacyImageBytesB64.IsEmpty();

    if (!bHasCanonicalImageBytes && !bHasLegacyImageBytes)
    {
        OutError = FString::Printf(
            TEXT("CAPTURE_NOW dispatch failed: sample manager observation missing image bytes (%s)."),
            *Target->GetName()
        );
        return false;
    }

    if (!ObservationPayload->TryGetStringField(TEXT("drone_id"), OutResolvedDroneId) || OutResolvedDroneId.IsEmpty())
    {
        OutResolvedDroneId = DroneId;
    }
    if (!ObservationPayload->TryGetStringField(TEXT("capture_id"), OutResolvedCaptureId) || OutResolvedCaptureId.IsEmpty())
    {
        OutResolvedCaptureId = CaptureId;
    }

    OutObservationPayload = ObservationPayload;
    return true;
}

void ASetDataConfigRuntimeActor::HandleActionMessage(
    const FString& MessageType,
    const FString& RunId,
    const TSharedPtr<FJsonObject>& Root,
    const TSharedPtr<FJsonObject>& PayloadObject
)
{
    if (!bConfigReady)
    {
        SendError(RunId, FString::Printf(TEXT("blocked pre-config-ready action: %s"), *MessageType));
        return;
    }

    if (MessageType.Equals(TEXT("SPAWN_DRONES"), ESearchCase::IgnoreCase))
    {
        HandleSpawnDrones(RunId, PayloadObject);
        return;
    }
    if (MessageType.Equals(TEXT("CMD"), ESearchCase::IgnoreCase))
    {
        HandleCmd(RunId, Root, PayloadObject);
        return;
    }
    if (MessageType.Equals(TEXT("CAPTURE_NOW"), ESearchCase::IgnoreCase))
    {
        HandleCaptureNow(RunId, Root, PayloadObject);
        return;
    }
}

void ASetDataConfigRuntimeActor::HandleSpawnDrones(const FString& RunId, const TSharedPtr<FJsonObject>& PayloadObject)
{
    UWorld* World = GetWorld();
    if (World == nullptr)
    {
        SendError(RunId, TEXT("World is null during SPAWN_DRONES."));
        return;
    }

    UClass* PawnClass = StaticLoadClass(APawn::StaticClass(), nullptr, *DronePawnClassPath);
    if (PawnClass == nullptr)
    {
        SendError(RunId, FString::Printf(TEXT("Failed to load drone pawn class: %s"), *DronePawnClassPath));
        return;
    }

    int32 SpawnCount = 1;
    double ParsedCount = 1.0;
    if (PayloadObject->TryGetNumberField(TEXT("spawn_count"), ParsedCount) || PayloadObject->TryGetNumberField(TEXT("count"), ParsedCount))
    {
        SpawnCount = FMath::Max(1, static_cast<int32>(ParsedCount));
    }

    TArray<FString> RequestedIds;
    const TArray<TSharedPtr<FJsonValue>>* IdArray = nullptr;
    if (PayloadObject->TryGetArrayField(TEXT("drone_ids"), IdArray) && IdArray != nullptr)
    {
        for (const TSharedPtr<FJsonValue>& Value : *IdArray)
        {
            FString IdText;
            if (Value.IsValid() && Value->TryGetString(IdText) && !IdText.IsEmpty())
            {
                RequestedIds.Add(IdText);
            }
        }
    }

    TArray<TSharedPtr<FJsonValue>> SpawnedIdValues;
    for (int32 Index = 0; Index < SpawnCount; ++Index)
    {
        const FString DroneId = (RequestedIds.IsValidIndex(Index) && !RequestedIds[Index].IsEmpty())
            ? RequestedIds[Index]
            : NextDroneId();
        const FVector SpawnLocation(static_cast<double>(Index) * SpawnSpacingCm, 0.0, 120.0);
        APawn* SpawnedPawn = World->SpawnActor<APawn>(PawnClass, SpawnLocation, FRotator::ZeroRotator);
        if (SpawnedPawn == nullptr)
        {
            continue;
        }
        SpawnedPawn->Tags.AddUnique(FName(*FString::Printf(TEXT("DroneId:%s"), *DroneId)));
        SpawnedDrones.Add(DroneId, SpawnedPawn);
        ApplySensorRigConfigToPawnIfAvailable(RunId, SpawnedPawn);
        SpawnedIdValues.Add(MakeShared<FJsonValueString>(DroneId));
    }

    TSharedPtr<FJsonObject> StatusPayload = MakeShared<FJsonObject>();
    StatusPayload->SetStringField(TEXT("event"), TEXT("SPAWN_DRONES_ACCEPTED"));
    StatusPayload->SetNumberField(TEXT("spawned_count"), SpawnedIdValues.Num());
    StatusPayload->SetArrayField(TEXT("drone_ids"), SpawnedIdValues);
    SendEnvelope(TEXT("STATUS"), RunId, StatusPayload);
}

void ASetDataConfigRuntimeActor::HandleCmd(
    const FString& RunId,
    const TSharedPtr<FJsonObject>& Root,
    const TSharedPtr<FJsonObject>& PayloadObject
)
{
    FString DroneId;
    if (!TryGetStringField(Root, TEXT("drone_id"), DroneId))
    {
        if (!TryGetStringField(PayloadObject, TEXT("drone_id"), DroneId))
        {
            SendError(RunId, TEXT("CMD missing drone_id."));
            return;
        }
    }

    APawn* Pawn = ResolveDroneById(DroneId);
    if (Pawn == nullptr)
    {
        SendError(RunId, FString::Printf(TEXT("CMD target not found: %s"), *DroneId));
        return;
    }

    const double Pitch = ReadNumber(PayloadObject, TEXT("pitch"), 0.0);
    const double Roll = ReadNumber(PayloadObject, TEXT("roll"), 0.0);
    const double Yaw = ReadNumber(PayloadObject, TEXT("yaw"), 0.0);
    const double Throttle = ReadNumber(PayloadObject, TEXT("throttle"), 0.0);

    const FVector TranslationDelta(
        Roll * CommandTranslationScaleCm,
        Pitch * CommandTranslationScaleCm,
        Throttle * CommandTranslationScaleCm
    );
    const FRotator RotationDelta(
        Pitch * CommandRotationScaleDeg,
        Yaw * CommandRotationScaleDeg,
        Roll * CommandRotationScaleDeg
    );

    Pawn->AddActorWorldOffset(TranslationDelta, false);
    Pawn->AddActorWorldRotation(RotationDelta);

    TSharedPtr<FJsonObject> StatusPayload = MakeShared<FJsonObject>();
    StatusPayload->SetStringField(TEXT("event"), TEXT("CMD_APPLIED"));
    StatusPayload->SetStringField(TEXT("drone_id"), DroneId);
    StatusPayload->SetNumberField(TEXT("pitch"), Pitch);
    StatusPayload->SetNumberField(TEXT("roll"), Roll);
    StatusPayload->SetNumberField(TEXT("yaw"), Yaw);
    StatusPayload->SetNumberField(TEXT("throttle"), Throttle);
    SendEnvelope(TEXT("STATUS"), RunId, StatusPayload, DroneId);
}

void ASetDataConfigRuntimeActor::HandleCaptureNow(
    const FString& RunId,
    const TSharedPtr<FJsonObject>& Root,
    const TSharedPtr<FJsonObject>& PayloadObject
)
{
    FString DroneId;
    if (!TryGetStringField(Root, TEXT("drone_id"), DroneId))
    {
        TryGetStringField(PayloadObject, TEXT("drone_id"), DroneId);
    }

    FString CaptureId;
    if (!TryGetStringField(Root, TEXT("capture_id"), CaptureId))
    {
        if (!TryGetStringField(PayloadObject, TEXT("capture_id"), CaptureId))
        {
            CaptureId = FString::Printf(TEXT("capture_%06d"), OutSeq);
        }
    }

    FString ResolveManagerError;
    ASampleManagerRuntimeActor* SampleManager = ResolveSampleManagerActor(ResolveManagerError);
    if (SampleManager == nullptr)
    {
        SendError(RunId, ResolveManagerError);
        return;
    }

    TSharedPtr<FJsonObject> ObsPayload;
    FString ResolvedDroneId;
    FString ResolvedCaptureId;
    FString CaptureError;
    if (!InvokeSampleManagerCaptureNow(
            SampleManager,
            RunId,
            DroneId,
            CaptureId,
            ObsPayload,
            ResolvedDroneId,
            ResolvedCaptureId,
            CaptureError
        ))
    {
        SendError(RunId, CaptureError);
        return;
    }

    const FString FinalDroneId = ResolvedDroneId.IsEmpty() ? DroneId : ResolvedDroneId;
    const FString FinalCaptureId = ResolvedCaptureId.IsEmpty() ? CaptureId : ResolvedCaptureId;
    SendEnvelope(TEXT("OBS"), RunId, ObsPayload, FinalDroneId, FinalCaptureId);
}

APawn* ASetDataConfigRuntimeActor::ResolveDroneById(const FString& DroneId) const
{
    if (!DroneId.IsEmpty())
    {
        if (const TWeakObjectPtr<APawn>* Found = SpawnedDrones.Find(DroneId))
        {
            if (Found->IsValid())
            {
                return Found->Get();
            }
        }
    }

    UWorld* World = GetWorld();
    if (World == nullptr)
    {
        return nullptr;
    }

    for (TActorIterator<APawn> It(World); It; ++It)
    {
        APawn* Candidate = *It;
        if (Candidate == nullptr)
        {
            continue;
        }
        if (DroneId.IsEmpty())
        {
            return Candidate;
        }
        const FString ExpectedTag = FString::Printf(TEXT("DroneId:%s"), *DroneId);
        for (const FName& Tag : Candidate->Tags)
        {
            if (Tag.ToString().Equals(ExpectedTag, ESearchCase::CaseSensitive))
            {
                return Candidate;
            }
        }
    }
    return nullptr;
}

FString ASetDataConfigRuntimeActor::NextDroneId() const
{
    int32 Index = 1;
    while (true)
    {
        const FString Candidate = FString::Printf(TEXT("drone_%04d"), Index);
        if (!SpawnedDrones.Contains(Candidate))
        {
            return Candidate;
        }
        ++Index;
    }
}
